import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.helper import gaussian_log_p, gaussian_sample

class Glow(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_block,
        in_seqlen
    ):
        super(Glow, self).__init__()

        self.flow_padding = nn.Parameter(torch.zeros(in_channel))
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        self.in_seqlen = in_seqlen
        affine = True

        for i in range(n_block - 1):
            if i == 0:
                i_squeeze = 1
                split=False
            else:
                i_squeeze = 2
                split=True
            self.in_seqlen //= i_squeeze
            self.blocks.append(Block(n_channel, n_flow, split=split, affine=affine, in_seqlen=self.in_seqlen, squeeze_size = i_squeeze))
            n_channel *= 1

        self.in_seqlen //= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine, in_seqlen=self.in_seqlen))

    def forward(self, x):
        logdet = 0
        log_sum = 0
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z, eps=None, reconstruct=False): 
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z[-1], z[-1], reconstruct=reconstruct)

            else:
                x = block.reverse(x, z[-(i + 1)], reconstruct=reconstruct)

        last_layer = x
        return x

class Block(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        split=True,
        affine=True, 
        in_seqlen=64, 
        squeeze_size=2
    ):
        super(Block, self).__init__()

        self.squeeze_size = squeeze_size
        squeeze_dim = in_channel * self.squeeze_size

        self.flows = nn.ModuleList() 
        for _ in range(n_flow):
            self.flows.append(FlowStep(squeeze_dim, in_seqlen, affine=affine))

        self.split = split
        if split:
            self.prior = ZeroLinear(in_channel, in_channel * 2)
        else:
            self.prior = ZeroLinear(in_channel * self.squeeze_size, in_channel * self.squeeze_size * 2)

    def forward(self, x):
        logdet = 0
        b_size, seq_len, n_channel = x.shape

        x_squeezed = x.view(b_size, seq_len // self.squeeze_size, self.squeeze_size, n_channel)
        x_squeezed = x_squeezed.permute(0, 1, 3, 2).contiguous()
        out = x_squeezed.view(b_size, seq_len // self.squeeze_size, n_channel * self.squeeze_size)
        
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_split = out.chunk(2, 1)
            mean, log_std = self.prior(out).chunk(2, 2)
            log_p = gaussian_log_p(z_split, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_std = self.prior(zero).chunk(2, 2)
            log_p = gaussian_log_p(out, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
            z_split = out

        return out, logdet, log_p, z_split

    def reverse(self, x, eps=None, reconstruct=False):
        input = x

        if reconstruct:
            if self.split:
                input = torch.cat([x, eps], 2)
            else:
                input = eps

        else:
            if self.split:
                mean, log_std = self.prior(input).chunk(2, 2)
                z = gaussian_sample(eps, mean, log_std)
                input = torch.cat([x, z], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_std = self.prior(zero).chunk(2, 2)
                z = gaussian_sample(eps, mean, log_std)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        
        b_size, seq_len, n_channel = input.shape

        input_unsqueezed = input.view(b_size, seq_len, n_channel // self.squeeze_size, self.squeeze_size)
        input_unsqueezed = input_unsqueezed.permute(0, 1, 3, 2).contiguous()
        input_unsqueezed = input_unsqueezed.view(b_size, seq_len * self.squeeze_size, n_channel // self.squeeze_size)
        return input_unsqueezed

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channel,
        in_seqlen,
        affine=True
    ):
        super(FlowStep, self).__init__()
        self.act_norm = ActNorm(in_channel, in_seqlen)
        self.inv_conv = Invertible1dConv(in_channel)
        self.affine_coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, x):
        x1, logdet = self.act_norm(x)
        x2, det1 = self.inv_conv(x1)
        x3, det2 = self.affine_coupling(x2)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return x3, logdet

    def reverse(self, x):
        x1 = self.affine_coupling.reverse(x)
        x2 = self.inv_conv.reverse(x1)
        x3 = self.act_norm.reverse(x2)

        return x3


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super(AffineCoupling, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroLinear(filter_size, in_channel)
        )

        self.affine = affine

    def forward(self, x):
        in_a, in_b = x.chunk(2, 1)        
        temp = self.net(in_a)   

        if self.affine:
            log_s, t = temp.chunk(2, 2)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(x.size(0), -1), 1)

        else:
            out_b = in_b + temp
            logdet = None
        output = torch.cat([in_a, out_b], 1)
        return output, logdet

    def reverse(self, x):
        out_a, out_b = x.chunk(2, 1)
        x_1 = conv_1(out_a)
        x_2 = relu(x_1)
        x_3 = conv_2(x_2)
        x_4 = relu(x_3)
        temp = convZero(x_4)
        log_s, t = temp.chunk(2, 2)
        s = torch.sigmoid(log_s + 2)
        in_b = out_b / s - t
        
        inputs = torch.cat([out_a, in_b], 1)
        return inputs

class Invertible1dConv(nn.Module):
    def __init__(self, in_channel):
        super(Invertible1dConv, self).__init__()

        w = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(w)
        self.weight = nn.Parameter(q)

    def forward(self, x):
        _, seq_length, _ = x.shape

        out = F.linear(x, self.weight)
        logdet = seq_length * torch.slogdet(self.weight)[1]

        return out, logdet

    def reverse(self, x):
        return F.linear(x, self.weight.inverse()) 

class ActNorm(nn.Module):
    def __init__(self, in_channel, len, logdet=True):
        super(ActNorm, self).__init__()
        
        self.loc = nn.Parameter(torch.zeros(1, in_channel, len))
        self.scale = nn.Parameter(torch.ones(1, in_channel, len))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad(): 
            flatten = x.permute(2, 0, 1).contiguous().view(x.shape[2], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            std = (
                flatten.std(1)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-12))
    
    def forward(self, x):
        _, seq_length, _ = x.shape
        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_abs = torch.log(torch.abs(self.scale))
        logdet = seq_length * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (x + self.loc), logdet
        else:
            return self.scale * (x + self.loc)

    def reverse(self, x):
        return x / self.scale - self.loc

class ZeroLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroLinear, self).__init__()

        self.linear = nn.Linear(in_channel, out_channel)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel))

    def forward(self, x):
        return self.linear(x) * torch.exp(self.scale * 3)




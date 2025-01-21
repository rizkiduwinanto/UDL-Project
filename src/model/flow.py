import torch
import torch.nn as nn

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

        for _ in range(n_block - 1):
            if i == 0:
                i_squeeze = 1
                split=False
            else:
                i_squeeze = 2
                split=True

            self.in_seqlen //= i_squeeze
            self.blocks.append(Block(n_channel, n_flow, split=split, in_seqlen=self.in_seqlen, squeeze_size = i_squeeze))
            n_channel *= 1

        self.in_seqlen //= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, in_seqlen=self.in_seqlen))

    def forward(self, x, length):
        logdet = 0
        out = x
        for block in self.blocks:
            out, det, _, _ = block(out, length)
            logdet = logdet + det

        return out, logdet

    def reverse(self, z, eps=None, reconstruct=False): 
        out = z
        for block in self.blocks[::-1]:
            out = block.reverse(out, eps, reconstruct)
        return out

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
            self.flows.append(FlowStep(squeeze_dim, in_seqlen))

        self.split = split
        if split:
            self.prior = ZeroLinear(in_channel, in_channel * 2)
        else:
            self.prior = ZeroLinear(in_channel * self.squeeze_size, in_channel * self.squeeze_size * 2)

    def forward(self, x, length):
        logdet = 0
        b_size, seq_len, n_channel = x.shape

        x = x.view(b_size, seq_length // self.squeeze_size, self.squeeze_size, n_channel)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b_size, seq_len // self.squeeze_size, n_channel * self.squeeze_size)
        
        for flow in self.flows:
            x, det = flow(x, length)
            logdet = logdet + det

        if self.split:
            x, z_split = x.chunk(2, 1)
            mean, log_std = self.prior(x).chunk(2, 2)
            log_p = gaussian_log_p(z_split, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(x)
            mean, log_std = self.prior(zero).chunk(2, 2)
            log_p = gaussian_log_p(x, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
            z_split = x

        return x, logdet, log_p, z_split

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
                z_split = gaussian_sample(eps, mean, log_std)
                input = torch.cat([x, z_split], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_std = self.prior(zero).chunk(2, 2)
                z_split = gaussian_sample(eps, mean, log_std)
                input = z_split

        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        
        b_size, seq_len, n_channel = input.shape

        input = input.view(b_size, seq_len, n_channel // self.squeeze_size, self.squeeze_size)
        input = input.permute(0, 1, 3, 2).contiguous()
        input = input.view(b_size, seq_len * self.squeeze_size, n_channel // self.squeeze_size)
        return input

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channel,
        in_seqlen
    ):
        super(Flow, self).__init__()
        self.act_norm = ActNorm(in_channel, in_seqlen)
        self.inv_conv = Invertible1dConv(in_channel)
        self.affine_coupling = AffineCoupling(in_channel)

    def forward(self, x):
        x, logdet = self.act_norm(x)
        x, logdet = self.inv_conv(x)
        x, logdet = self.affine_coupling(x)

        return x, logdet

    def reverse(self, x):
        x = self.affine_coupling.reverse(x)
        x = self.inv_conv.reverse(x)
        x = self.act_norm.reverse(x)

        return x


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super(AffineCoupling, self).__init__()

        conv_1 = nn.Conv1d(in_channel // 2, filter_size, 3, padding=1)
        conv_2 = nn.Conv1d(filter_size, filter_size, 1)
        relu = nn.ReLU(inplace=True)
        convZero - ZeroLinear(filter_size, in_channel)

    def forward(self, x):
        in_a, in_b = x.chunk(2, 1)
        x_1 = conv_1(in_a)
        x_2 = relu(x_1)
        x_3 = conv_2(x_2)
        x_4 = relu(x_3)
        temp = convZero(x_4)
        log_s, t = temp.chunk(2, 2)
        s = torch.sigmoid(log_s + 2)
        out_b = (in_b + t) * s

        logdet = torch.sum(torch.log(s).view(x.size(0), -1), 1)

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
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)
        logdet = seq_length * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, x):
        return x / self.scale - self.loc

class Invertible1dConv(nn.Module):
    def __init__(self, in_channel):
        super(Invertible1dConv, self).__init__()

        w = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(w)
        self.weight = nn.Parameter(q)

    def forward(self, x):
        _, seq_length, _ = x.shape

        out = F.linear(x, self.weight)
        logdet = seq_length * torch.slogdet(self.weight)[1]

        return out, logdet

    def reverse(self, x):
        return F.linear(x, self.weight.inverse()) 
    

class ZeroLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroLinear, self).__init__()

        self.linear = nn.Linear(in_channel, out_channel)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero()
        self.scale = nn.Parameter(torch.zeros(1, out_channel))

    def forward(self, x):
        return self.linear(x) * torch.exp(self.scale * 3)




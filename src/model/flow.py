import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.helper import gaussian_log_p, gaussian_sample

class Glow(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_block
    ):
        super(Glow, self).__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        affine = True

        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, x):
        logdet = 0
        log_sum = 0
        out = x
        z_outs = []

        print("Glow shape:", out.shape)

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
        return x

class Block(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        split=True,
        affine=True, 
        squeeze_size=4
    ):
        super(Block, self).__init__()

        self.squeeze_size = squeeze_size
        squeeze_dim = in_channel * self.squeeze_size

        self.flows = nn.ModuleList() 
        for _ in range(n_flow):
            self.flows.append(FlowStep(squeeze_dim, affine=affine))

        self.split = split
        if split:
            self.prior = ZeroLinear(in_channel, in_channel * 2)
        else:
            self.prior = ZeroLinear(in_channel * self.squeeze_size, in_channel * self.squeeze_size * 2)

    def forward(self, x):
        logdet = 0
        print("Block shape:", x.shape)
        b_size, n_channel, seq_len = x.shape

        x_squeezed = x.view(b_size, n_channel, seq_len // self.squeeze_size, self.squeeze_size)
        x_squeezed = x_squeezed.permute(0, 1, 3, 2).contiguous()
        print("Block squeezed shape:", x_squeezed.shape)
        out = x_squeezed.view(b_size, n_channel * self.squeeze_size, seq_len // self.squeeze_size)
        
        print("Block out shape:", out.shape)
        
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
        affine=True
    ):
        super(FlowStep, self).__init__()
        self.act_norm = ActNorm(in_channel)
        self.inv_conv = Invertible1dConv(in_channel)
        self.affine_coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, x):
        print("FlowStep shape:", x.shape)
        x1, logdet = self.act_norm(x)
        
        print("FlowStep act norm shape:", x1.shape)
        x2, det1 = self.inv_conv(x1)
        
        print("FlowStep conv shape:", x2.shape)
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
    def __init__(self, in_channel, filter_size=192, affine=True):
        super(AffineCoupling, self).__init__()
        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroLinear(filter_size, in_channel if self.affine else in_channel // 2)
        )

    def forward(self, x):
        print("AffineCoupling input shape:", x.shape)
        
        # Split input along channel dimension
        in_a, in_b = x.chunk(2, 1)
        print("AffineCoupling chunk shapes:", in_a.shape, in_b.shape)
        
        temp = self.net(in_a)
        print("AffineCoupling temp shape:", temp.shape)
        
        if self.affine:
            log_s, t = temp.chunk(2, 2)
            print("AffineCoupling log_s shape:", log_s.shape)
            print("AffineCoupling t shape:", t.shape)
            
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(x.size(0), -1), 1)
        else:
            out_b = in_b + temp
            logdet = None
        
        # Combine results
        output = torch.cat([in_a, out_b], dim=1)
        print("AffineCoupling output shape:", output.shape)
        
        return output, logdet

    def reverse(self, x):
        out_a, out_b = x.chunk(2, 1)
        temp = self.net(out_a)
        log_s, t = x.chunk(2, 2)
        s = torch.sigmoid(log_s + 2)
        in_b = out_b / s - t
        
        inputs = torch.cat([out_a, in_b], dim=1)
        return inputs

class Invertible1dConv(nn.Module):
    def __init__(self, in_channel):
        super(Invertible1dConv, self).__init__()

        w = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(w)
        weight = q.unsqueeze(2)
        print("Invertible1dConv weight shape:", weight.shape)
        self.weight = nn.Parameter(q)

    def forward(self, x):
        print("Invertible1dConv shape:", x.shape)
        _, _, seq_length = x.shape

        out = F.linear(x.transpose(1, 2), self.weight).transpose(1, 2)
        if torch.device == "cuda":
            logdet = seq_length * torch.slogdet(self.weight.double())[1].float()
        else:
            logdet = seq_length * torch.slogdet(self.weight.float())[1].float()
        
        return out, logdet

    def reverse(self, x):
        return F.linear(x.transpose(1, 2), self.weight.inverse()).transpose(1, 2)

class ActNorm(nn.Module):
    def __init__(self, num_channels, logdet=True):
        super(ActNorm, self).__init__()
        
        self.loc = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad(): 
            mean = x.mean(dim=[0, 2], keepdim=True)
            std = x.std(dim=[0, 2], keepdim=True)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-12))
    
    def forward(self, x):
        _, _, seq_length = x.shape
        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_abs = torch.log(torch.abs(self.scale)).sum(dim=1).unsqueeze(1)
        logdet = seq_length * log_abs

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
        x_reshape = x.view(x.size(0), -1)
        print("ZeroLinear shape:", x_reshape.shape)
        print("ZeroLinear weight shape:", self.linear.weight.shape)
        print("ZeroLinear scale shape:", self.scale.shape)
        out = self.linear(x) * torch.exp(self.scale * 3)
        return out.view(x.size(0), -1, x.size(2))




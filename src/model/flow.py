import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy import linalg as la
import numpy as np

from utils.helper import gaussian_log_p, gaussian_sample

class Glow(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_block,
        affine=True,
        conv_lu=True,
        device= "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(Glow, self).__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        affine = affine

        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, device=device))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine, device=device))

    def forward(self, x):
        logdet = 0
        log_p_sum = 0
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p


        return log_p_sum, logdet, z_outs

    def reverse(self, z, reconstruct=False): 
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
        conv_lu=True,
        squeeze_size=4,
        device="cuda"
    ):
        super(Block, self).__init__()

        self.squeeze_size = squeeze_size
        squeeze_dim = in_channel * self.squeeze_size

        self.device = device

        self.flows = nn.ModuleList() 
        for _ in range(n_flow):
            self.flows.append(FlowStep(squeeze_dim, affine=affine, conv_lu=conv_lu, device=device))

        self.split = split
        if split:
            self.prior = ZeroConv2d(squeeze_dim // 2, squeeze_dim).to(self.device)
        else:
            self.prior = ZeroConv2d(squeeze_dim, squeeze_dim * 2).to(self.device)

    def forward(self, x):
        logdet = 0
        b_size, n_channel, h, w = x.shape

        x_squeezed = x.view(b_size, n_channel, h // 2, 2, w // 2, 2)
        x_squeezed = x_squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
        out = x_squeezed.view(b_size, n_channel * self.squeeze_size, h // 2, w // 2)

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_split = out.chunk(2, 1)
            mean, log_std = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_split, mean, log_std)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out).to(self.device)
            mean, log_std = self.prior(zero).chunk(2, 1)
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
                mean, log_std = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                input = torch.cat([x, z], 1)
            else:
                zero = torch.zeros_like(input).to(self.device)
                mean, log_std = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        
        b_size, n_channel, h, w = input.shape

        input_unsqueezed = input.view(b_size, n_channel // self.squeeze_size, 2, 2, h, w)
        input_unsqueezed = input_unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        input_unsqueezed = input_unsqueezed.view(b_size, n_channel // self.squeeze_size, h * 2, w * 2)

        return input_unsqueezed
  
class FlowStep(nn.Module):
    def __init__(
        self,
        in_channel,
        affine=True,
        conv_lu=True,
        device="cuda"
    ):
        super(FlowStep, self).__init__()
        self.act_norm = ActNorm(in_channel, device=device)

        self.device = device

        if conv_lu:
            self.inv_conv = Invertible2dConvLU(in_channel, device=device)
        else:
            self.inv_conv = Invertible2dConv(in_channel, device=device)

        self.affine_coupling = AffineCoupling(in_channel, affine=affine, device=device)

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
    def __init__(self, in_channel, filter_size=512, affine=True, device="cuda"):
        super(AffineCoupling, self).__init__()
        self.affine = affine
        self.device = device

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        ).to(self.device)

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        in_a, in_b = x.chunk(2, 1)
        temp = self.net(in_a.to(self.device))
        
        if self.affine:
            log_s, t = temp.chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(x.size(0), -1), 1)
        else:
            out_b = in_b + temp
            logdet = None
        
        output = torch.cat([in_a, out_b], dim=1)
        return output, logdet.to(self.device)

    def reverse(self, x):
        out_a, out_b = x.chunk(2, 1)

        if self.affine:
            temp = self.net(out_a)
            log_s, t = temp.chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            in_b = out_b - self.net(out_a)

        inputs = torch.cat([out_a, in_b], dim=1)
        return inputs


# class Transformer(nn.Module):
#     def __init__(self, in_channel, device="cuda"):
#         super(Transformer, self).__init__()

#         self.device = device

#         self.net = nn.Sequential(
#             nn.Conv2d(in_channel, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, in_channel, 3, padding=1),
#         ).to(self.


class Invertible2dConv(nn.Module):
    def __init__(self, in_channel, device="cuda"):
        super(Invertible2dConv, self).__init__()

        self.device = device

        w = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(w)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(q)

    def forward(self, x):
        _, _, h, w = x.shape

        out = F.conv2d(x.to(self.device), self.weight)
        logdet = h * w * torch.slogdet(self.weight.squeeze().double())[1].float()

        return out, logdet.to(self.device)

    def reverse(self, x):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )

class Invertible2dConvLU(nn.Module):
    def __init__(self, in_channel, device="cuda"):
        super(Invertible2dConvLU, self).__init__()

        self.device = device

        w = torch.randn(in_channel, in_channel)
        q, _ = la.qr(w)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        _, _, h, w = x.shape

        weight = self.calc_weight()

        out = F.conv2d(x.to(self.device), weight.to(self.device))
        logdet = h * w * torch.sum(self.w_s)

        return out, logdet.to(self.device)

    def reverse(self, x):
        weight = self.calc_weight()

        return F.conv2d(x.to(self.device), weight.squeeze().inverse().unsqueeze(2).unsqueeze(3).to(self.device))

class ActNorm(nn.Module):
    def __init__(self, num_channels, logdet=True, device="cuda"):
        super(ActNorm, self).__init__()
        
        self.loc = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))

        self.device = device

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad(): 
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            std = (
                flatten.std(1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            self.loc.data.copy_(-mean.to(self.device))
            self.scale.data.copy_(1 / (std + 1e-6).to(self.device))
    
    def forward(self, x):
        _, _, h, w = x.shape

        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_abs = torch.log(torch.abs(self.scale)).sum(dim=1).unsqueeze(1)
        logdet = h * w * log_abs

        if self.logdet:
            return self.scale.to(self.device) * (x.to(self.device) + self.loc.to(self.device)), logdet.to(self.device)
        else:
            return self.scale.to(self.device) * (x.to(self.device) + self.loc.to(self.device))

    def reverse(self, x):
        return x / self.scale.to(self.device) - self.loc.to(self.device)

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out




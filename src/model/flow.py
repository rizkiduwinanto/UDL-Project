import torch
import torch.nn as nn

class Glow(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_block,
        in_seqlen
    ):
        super(Glow, self).__init__()

        self.flow = Flow(in_channel, n_flow, n_block, in_seqlen)


class Block(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_block,
        in_seqlen
    ):
        super(Block, self).__init__()

        self.flow = Flow(in_channel, n_flow, n_block, in_seqlen)

    def forward(self, x):
        pass

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_block,
        in_seqlen
    ):
        super(Flow, self).__init__()
        self.act_norm = ActNorm(in_channel, in_seqlen)
        self.inv_conv = Invertible1dConv(in_channel)
        self.affine_coupling = AffineCoupling(in_channel)


    def forward(self, x):
        pass


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, features):
        conv_1 = nn.Conv(features)
        conv_2 = nn.Conv(features)
        relu = nn.ReLU()

    def forward(self, x):
        pass

    def reverse(self, x):
        pass
        


class ActNorm(nn.Module):
    def __init__(self, in_channel, len, logdet=True):
        super(ActNorm, self).__init__()
        
        self.loc = nn.Parameter(torch.zeros(1, in_channel, len))
        self.scale = nn.Parameter(torch.ones(1, in_channel, len))

    def forward(self, x):
        pass

    def reverse(self, x):
        pass

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




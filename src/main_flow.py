import torch
from model.flow import Glow

if __name__ == '__main__':
    data = torch.randn(2, 32, 32)
    print(data.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    glow = Glow(in_channel=1, n_flow=2, n_block=2, device=device)
    log_p_sum, logdet, z_outs = glow.forward(data.unsqueeze(1))
    print(log_p_sum.shape, logdet.shape, z_outs[0].shape)

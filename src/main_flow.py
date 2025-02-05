import torch
from model.flow import Glow

if __name__ == '__main__':
    data = torch.randn(2, 32, 32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    glow = Glow(in_channel=1, n_flow=100, n_block=4, device=device)
    log_p_sum, logdet, z_outs = glow.forward(data.unsqueeze(1))

    n_block = 4
    batch_size = 2
    latent_channels = 1
    latent_height = 32
    latent_width = 32 
    z_list = []
    for i in range(n_block - 1):
        latent_channels *= 2
        latent_height //= 2
        latent_width //= 2
        z = torch.randn(batch_size, latent_channels, latent_height, latent_width).to(device)
        z_list.append(z)
    latent_height //= 2
    latent_width //= 2
    z = torch.randn(batch_size, latent_channels * 4, latent_height, latent_width).to(device)
    z_list.append(z)

    x = glow.reverse(z_list).cpu()
    print(x.shape)

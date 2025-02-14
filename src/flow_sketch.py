import torch
from model.flow import Glow

if __name__ == '__main__':
    data = torch.randn(2, 32, 32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    glow = Glow(in_channel=1, n_flow=100, n_block=4, device=device)
    
    # Forward pass
    log_p_sum, logdet, z_outs = glow.forward(data.unsqueeze(1))
    print("Forward pass outputs:")
    print(f"log_p_sum: {log_p_sum.shape}, logdet: {logdet.shape}, z_outs shapes: {[z.shape for z in z_outs]}")

    # Reverse pass with reconstruct=False
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
        print(f"Block {i + 1} latent shape: {z.shape}")
        z_list.append(z)
    latent_height //= 2
    latent_width //= 2
    z = torch.randn(batch_size, latent_channels * 4, latent_height, latent_width).to(device)
    z_list.append(z)

    x = glow.reverse(z_list, reconstruct=False).cpu()
    print("\nReverse pass with reconstruct=False:")
    print(f"Output shape: {x.shape}")

    # Reverse pass with reconstruct=True
    noise = [torch.randn_like(z) for z in z_outs]
    print(f"\nNoise shapes: {[n.shape for n in noise]}")
    x_recon = glow.reverse(noise, reconstruct=True).cpu()
    print("\nReverse pass with reconstruct=True:")
    print(f"Output shape: {x_recon.shape}")
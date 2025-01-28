import torch


def get_sample(self, flow_z, temperature=0.1, show_size=5):
    z_sample = []
    for z in flow_z:
        z_new = torch.empty(z.size()).normal_(mean=0,std=1.15)
        z_sample.append(z_new[:show_size].cuda())
    return z_sample

def sample_flow(
    model,
    flow_z=None,
    reconstruct=False,
    device="cuda",
    seq_len=384
):
    if 
    return embedding
import torch

pkl = "checkpoints/psp_ffhq_encode.pt"
data = torch.load(pkl) ##load_state_dict(data["g_ema"])
print(data['state_dict'].keys())
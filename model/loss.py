import torch
import torch.nn as nn

def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, embedding1, embedding2, embedding1_resid, embedding2_resid):
        cosine_loss = torch.mean(torch.abs(
            cosine_sim(embedding1.float(), embedding1_resid.float()) \
            + cosine_sim(embedding2.float(), embedding2_resid.float())
            ))
        return cosine_loss
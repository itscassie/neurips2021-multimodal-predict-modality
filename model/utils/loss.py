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

    def forward(self, emb1, emb2, emb1_resid, emb2_resid):
        emb1, emb2 = emb1.float(), emb2.float()
        cosine_loss = torch.mean(torch.abs(
            cosine_sim(emb1, emb1_resid) + cosine_sim(emb2, emb2_resid)
            ))
        return cosine_loss

class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss
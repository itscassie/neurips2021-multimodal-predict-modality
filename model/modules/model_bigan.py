import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

# 1. Change AE dropout rate
# 2. Change reconstruction loss to BCE
# 3. Larger G,D weights
# 4. Cluster head regularization
# 5. Cluster head gradient explosion problem
# 6. G / D balance problem

def _one_hot(tensor, num):
    b = list(tensor.size())[0]
    onehot = torch.cuda.FloatTensor(b, num).fill_(0)
    ones = torch.cuda.FloatTensor(b, num).fill_(1)
    out = onehot.scatter_(1, torch.unsqueeze(tensor, 1), ones)
    return out

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=1000):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=1000):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = x + torch.normal(mean=0, std=0.3, size=x.shape).to(x.device)
        return self.dis(x).view(-1)

class BiGAN(nn.Module):
    def __init__(self, input_dim, feat_dim, hidden_dim=1000):
        super(BiGAN, self).__init__()

        self.encoder = Encoder(input_dim, feat_dim, hidden_dim)
        self.decoder = Decoder(feat_dim, input_dim, hidden_dim)
        self.discriminator = Discriminator(feat_dim + input_dim, hidden_dim)


    def forward(self, x, z_real):
        """
            x: data
            z: real z
            c: real c
        """

        x = x.float()
        z_real = z_real.float()

        z_fake = self.encoder(x)
        x_fake = self.decoder(z_fake)
        
        real_d_in = torch.cat([x, z_real], dim=-1)
        fake_d_in = torch.cat([x_fake, z_fake], dim=-1)

        fake_score = self.discriminator(fake_d_in)
        real_score = self.discriminator(real_d_in)

        return x_fake, z_fake, fake_score, real_score

if __name__ == "__main__":
    z_dim = 15
    c_dim = 20
    hidden = 1000
    input_dim = 1000
    bsz = 5

    bigan = BiGAN(input_dim, z_dim, hidden).cuda()
    x = torch.randn(bsz, input_dim).cuda()
    z = torch.randn(bsz, z_dim).cuda()
    bigan(x, z)
    print(bigan)

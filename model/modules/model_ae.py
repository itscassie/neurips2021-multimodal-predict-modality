import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x_rec = self.decoder(x)
        return x_rec

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)
    
    def forward(self, x):
        feat = self.encoder(x)
        x_rec = self.decoder(feat)
        return x_rec

class UnbAutoEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, feat_dim, hid_dim_en, hid_dim_de):
        super(UnbAutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hid_dim_en)
        self.decoder = Decoder(feat_dim, out_dim, hid_dim_de)
    
    def forward(self, x):
        feat = self.encoder(x)
        x_rec = self.decoder(feat)
        return x_rec

class BN_common_encoder(nn.Module):
    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(BN_common_encoder, self).__init__()
                 
        self.rna_encoder = Encoder(rna_input_size, out_dim, hidden_dim)                         
        self.atac_encoder = Encoder(atac_input_size, out_dim, hidden_dim)
        self.rna_bn = nn.BatchNorm1d(rna_input_size, affine=True)
        self.atac_bn = nn.BatchNorm1d(atac_input_size, affine=True)

    def forward(self, rna_data, atac_data):
        rna_bn = self.rna_bn(rna_data)
        rna_cm_emb = self.rna_encoder(rna_bn)

        atac_bn = self.atac_bn(atac_data)
        atac_cm_emb = self.atac_encoder(atac_bn)

        return rna_cm_emb, atac_cm_emb

class BN_resid_encoder(nn.Module):
    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(BN_resid_encoder, self).__init__()
                 
        self.rna_encoder = Encoder(rna_input_size, out_dim, hidden_dim)                         
        self.atac_encoder = Encoder(atac_input_size, out_dim, hidden_dim)
        self.rna_bn = nn.BatchNorm1d(rna_input_size, affine=True)
        self.atac_bn = nn.BatchNorm1d(atac_input_size, affine=True)
                       
        
    def forward(self, rna_data, atac_data):
        rna_bn = self.rna_bn(rna_data)
        rna_resid_emb = self.rna_encoder(rna_bn)

        atac_bn = self.atac_bn(atac_data)
        atac_resid_emb = self.atac_encoder(atac_bn)
        
        return rna_resid_emb, atac_resid_emb
        

class BN_concat_decoder(nn.Module):
    def __init__(self, input_dim, rna_output_size, atac_output_size, hidden_dim=1000):
        super(BN_concat_decoder, self).__init__()              
        self.rna_decoder = Decoder(input_dim, rna_output_size, hidden_dim)                         
        self.atac_decoder = Decoder(input_dim, atac_output_size, hidden_dim)

    def forward(self, rna_cm_emb, atac_cm_emb, rna_resid_emb, atac_resid_emb):
        rna_embedding = rna_cm_emb + rna_resid_emb
        atac_embedding = atac_cm_emb + atac_resid_emb

        rna_rec = self.rna_decoder(rna_embedding)
        atac_rec = self.atac_decoder(atac_embedding)
                        
        return rna_rec, atac_rec


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
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = x + torch.normal(mean=0, std=0.3, size=x.shape).cuda()
        return self.dis(x).view(-1)

class Pix2Pix(nn.Module):
    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim=1000):
        super(Pix2Pix, self).__init__()

        self.autoencoder = AutoEncoder(input_dim, out_dim, feat_dim, hidden_dim)
        self.discriminator = Discriminator(out_dim, hidden_dim)

    def forward(self, x, y_real):

        y_fake = self.autoencoder(x)
        fake_score = self.discriminator(y_fake)
        real_score = self.discriminator(y_real)

        return y_fake, fake_score, real_score

if __name__ == "__main__":
    import torch
    bsz = 5
    input_dim = 100
    out_dim = 134 
    feat_dim = 64 
    hidden_dim = 1000

    pix2pix = Pix2Pix(input_dim, out_dim, feat_dim).cuda().float()
    x = torch.randn(bsz, input_dim).cuda()
    y_fake, fake_score, real_score = pix2pix(x)
    print(pix2pix)
    print(y_fake, fake_score, real_score)

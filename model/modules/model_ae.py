""" autoencoder based models """
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """base encoder module"""

    def __init__(self, input_dim, out_dim, hidden_dim, dropout=0.2):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x_input):
        """forward propogation of the encoder arch"""
        x_emb = self.encoder(x_input)
        return x_emb


class Decoder(nn.Module):
    """base decoder module"""

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
            nn.ReLU(),
        )

    def forward(self, x_emb):
        """forward propogation of the decoder arch"""
        x_rec = self.decoder(x_emb)
        return x_rec


class AutoEncoder(nn.Module):
    """autoencoder module"""

    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim, dropout=0.2):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim, dropout)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)

    def forward(self, x_input):
        """forward propogation of the autoencoder arch"""
        x_emb = self.encoder(x_input)
        x_rec = self.decoder(x_emb)
        return x_rec


class Discriminator(nn.Module):
    """base discriminator class for pix2pix method"""

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

    def forward(self, x_feat):
        """forward propogation of the discriminator arch"""
        x_feat = x_feat + torch.normal(mean=0, std=0.3, size=x_feat.shape).cuda()
        return self.dis(x_feat).view(-1)


class Pix2Pix(nn.Module):
    """pix2pix module"""

    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim=1000):
        super(Pix2Pix, self).__init__()

        self.autoencoder = AutoEncoder(input_dim, out_dim, feat_dim, hidden_dim)
        self.discriminator = Discriminator(out_dim, hidden_dim)

    def forward(self, x_input, y_real):
        """forward propogation of the pix2pix arch"""
        y_fake = self.autoencoder(x_input)
        fake_score = self.discriminator(y_fake)
        real_score = self.discriminator(y_real)

        return y_fake, fake_score, real_score


class BatchClassifier(nn.Module):
    """base batch classifier class"""

    def __init__(self, input_dim, cls_num=6, hidden_dim=50):
        super(BatchClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, cls_num),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x_feat):
        """forward propogation of the batch classifier arch"""
        return self.classifier(x_feat)


class BatchRemovalGAN(nn.Module):
    """batch removal module"""

    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim, cls_num=10, dropout=0.2):
        super(BatchRemovalGAN, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim, dropout)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)
        self.classifier = BatchClassifier(feat_dim, cls_num=cls_num)

    def forward(self, x_input):
        """forward propogation of the batch removal gan arch"""
        x_feat = self.encoder(x_input)
        x_rec = self.decoder(x_feat)
        cls_prob = self.classifier(x_feat)

        return x_rec, cls_prob


class CommonEncoder(nn.Module):
    """base common encoder for residual method"""

    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(CommonEncoder, self).__init__()

        self.rna_encoder = Encoder(rna_input_size, out_dim, hidden_dim)
        self.atac_encoder = Encoder(atac_input_size, out_dim, hidden_dim)
        self.rna_bn = nn.BatchNorm1d(rna_input_size, affine=True)
        self.atac_bn = nn.BatchNorm1d(atac_input_size, affine=True)

    def forward(self, rna_data, atac_data):
        """forward propogation of the common encoder arch"""
        rna_bn = self.rna_bn(rna_data)
        rna_cm_emb = self.rna_encoder(rna_bn)

        atac_bn = self.atac_bn(atac_data)
        atac_cm_emb = self.atac_encoder(atac_bn)

        return rna_cm_emb, atac_cm_emb


class ResidEncoder(nn.Module):
    """base residual encoder for residual method"""

    def __init__(self, rna_input_size, atac_input_size, out_dim, hidden_dim=1000):
        super(ResidEncoder, self).__init__()

        self.rna_encoder = Encoder(rna_input_size, out_dim, hidden_dim)
        self.atac_encoder = Encoder(atac_input_size, out_dim, hidden_dim)
        self.rna_bn = nn.BatchNorm1d(rna_input_size, affine=True)
        self.atac_bn = nn.BatchNorm1d(atac_input_size, affine=True)

    def forward(self, rna_data, atac_data):
        """forward propogation of the residual encoder arch"""
        rna_bn = self.rna_bn(rna_data)
        rna_resid_emb = self.rna_encoder(rna_bn)

        atac_bn = self.atac_bn(atac_data)
        atac_resid_emb = self.atac_encoder(atac_bn)

        return rna_resid_emb, atac_resid_emb


class ConcatEncoder(nn.Module):
    """base concat decoder for residual method"""

    def __init__(self, input_dim, rna_output_size, atac_output_size, hidden_dim=1000):
        super(ConcatEncoder, self).__init__()
        self.rna_decoder = Decoder(input_dim, rna_output_size, hidden_dim)
        self.atac_decoder = Decoder(input_dim, atac_output_size, hidden_dim)

    def forward(self, rna_cm_emb, atac_cm_emb, rna_resid_emb, atac_resid_emb):
        """forward propogation of the concat decoder arch"""
        rna_embedding = rna_cm_emb + rna_resid_emb
        atac_embedding = atac_cm_emb + atac_resid_emb

        rna_rec = self.rna_decoder(rna_embedding)
        atac_rec = self.atac_decoder(atac_embedding)

        return rna_rec, atac_rec


if __name__ == "__main__":

    bsz = 5
    in_d = 10
    out_d = 3
    feat_d = 2
    hid_d = 10

    x1 = torch.randn(bsz, in_d).cuda()

    model = AutoEncoder(in_d, out_d, feat_d, hid_d).cuda().float()
    print(model)
    output = model(x1)
    print(output.shape)

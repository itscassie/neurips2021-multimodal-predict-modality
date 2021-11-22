import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
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

class Decoder_onelayer(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Decoder_onelayer, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x):
        return self.decoder(x)       

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim, dropout=0.2):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim, dropout)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)

    def forward(self, x):
        feat = self.encoder(x)
        x_rec = self.decoder(feat)
        return x_rec


class Distribution_Kernel(nn.Module):
    def __init__(self, input_dim):
        super(Distribution_Kernel, self).__init__()
        self.kernel = nn.Sequential(
            nn.Conv1d(2, 16, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 1)
        )    
        
        self.batchnorm = nn.Sequential(
            nn.BatchNorm1d(input_dim, affine = True)
        ) 
    
    def forward(self, x):
        x = self.kernel(x)
        x = torch.squeeze(x, 1)
        return self.batchnorm(x)

class KernelAE(nn.Module):
    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim):
        super(KernelAE, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)  
        self.decoder2 = Decoder_onelayer(feat_dim, out_dim)  
        
        self.index_bias = Variable(torch.zeros(1, 1, out_dim), requires_grad=True).cuda()
        self.kernel = Distribution_Kernel(out_dim)
    
    def forward(self, x):        
        feat = self.encoder(x) 
        reconstruct_index = torch.unsqueeze(self.decoder(feat), 1)
        reconstruct_index2 = torch.unsqueeze(self.decoder2(feat), 1)
        expand_bias = self.index_bias.repeat(x.shape[0], 1, 1)

        reconstruct_index = torch.cat([reconstruct_index, reconstruct_index2], dim=1)
        reconstruct = self.kernel(reconstruct_index)
        
        return reconstruct

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

class BatchClassifier(nn.Module):
    def __init__(self, input_dim, cls_num=6, hidden_dim=50):
        super(BatchClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, cls_num),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = x# + torch.normal(mean=0, std=0.3, size=x.shape).to(device=device)
        return self.classifier(x)

class BatchRemovalGAN(nn.Module):
    def __init__(self, input_dim, out_dim, feat_dim, hidden_dim, cls_num=10, dropout=0.2):
        super(BatchRemovalGAN, self).__init__()
        self.encoder = Encoder(input_dim, feat_dim, hidden_dim, dropout)
        self.decoder = Decoder(feat_dim, out_dim, hidden_dim)
        self.classifier = BatchClassifier(feat_dim, cls_num=cls_num)

    def forward(self, x):
        feat = self.encoder(x)
        x_rec = self.decoder(feat)
        cls_prob = self.classifier(feat)
        
        return x_rec, cls_prob


class ModelEnsemble(nn.Module):
    def __init__(self, modelA, modelB, out_dim, device):
        super(ModelEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.w1 = torch.rand(out_dim, requires_grad=True, device=device, dtype=torch.float)
        self.w2 = torch.rand(1, requires_grad=True, device=device, dtype=torch.float)

    def forward(self, x1, x2):
        out1 = self.modelA(x1)
        out2 = self.modelB(x2)
        # out = out1 * self.w1 + out2 * (1 - self.w1)
        out = out1 * self.w2 + out2 * (1 - self.w2)
        return out

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

if __name__ == "__main__":
    import torch
    bsz = 5
    input_dim = 10
    out_dim = 3
    feat_dim = 2 
    hidden_dim = 10

    # pix2pix = Pix2Pix(input_dim, out_dim, feat_dim).cuda().float()
    # y_fake, fake_score, real_score = pix2pix(x)
    # print(pix2pix)
    # print(y_fake, fake_score, real_score)
    x1, x2 = torch.randn(bsz, input_dim).cuda(), torch.randn(bsz, input_dim).cuda()

    modelA = AutoEncoder(input_dim, out_dim, feat_dim, hidden_dim).cuda().float()
    modelB = AutoEncoder(input_dim, out_dim, feat_dim, hidden_dim).cuda().float()
    model = ModelEnsemble(modelA, modelB, out_dim, device=torch.device('cuda:0')).cuda().float()
    print(model)
    output = model(x1, x2)
    print(output.shape)

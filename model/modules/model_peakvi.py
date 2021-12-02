"""
The SCVI module contains simplified code from the python scvi package.
https://github.com/YosefLab/scvi-tools/
"""
from typing import Callable, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from modules.model_ae import AutoEncoder


def reparameterize_gaussian(mu, var):
    """
    mu: mean from the encoder's latent space
    var: variance from the encoder's latent space
    """
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    """return x"""
    return x


class Encoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
    ):

        super(Encoder, self).__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = nn.Sequential(
            # layer_0
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            # layer_1
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor):
        """
        The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = torch.clamp(self.mean_encoder(q), max=12)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent


class Decoder(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    """

    def __init__(self, n_input: int, n_output: int, n_hidden: int = 128):
        super(Decoder, self).__init__()
        self.px_decoder = nn.Sequential(
            # layer_0
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.LeakyReLU(negative_slope=0.01),
            # layer_1
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden, elementwise_affine=False),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid())

    def forward(self, z: torch.Tensor):
        """runs decoder"""
        x = self.output(self.px_decoder(z))
        return x


class PEAKVAE(nn.Module):
    """
    Variational auto-encoder model for ATAC-seq data.
    This is an implementation of the peakVI model descibed in.
    Parameters
    ----------
    n_input_regions
        Number of input regions.
    n_hidden
        Number of nodes per hidden layer. If `None`, defaults to square root
        of number of regions.
    n_latent
        Dimensionality of the latent space. If `None`, defaults to square root
        of `n_hidden`.
    dropout_rate
        Dropout rate for neural networks
    model_depth
        Model library size factors or not.
    region_factors
        Include region-specific factors in the model
    latent_distribution
        which latent distribution to use, options are
        * ``'normal'`` - Normal distribution (default)
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    """

    def __init__(
        self,
        n_input_regions: int,
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = None,
        dropout_rate: float = 0.1,
        model_depth: bool = True,
        region_factors: bool = True,
        latent_distribution: str = "normal",
    ):
        super().__init__()

        self.n_input_regions = n_input_regions
        self.n_hidden = int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.model_depth = model_depth
        self.dropout_rate = dropout_rate
        self.latent_distribution = latent_distribution

        n_input_encoder = self.n_input_regions
        self.z_encoder = Encoder(
            n_input=n_input_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            distribution=self.latent_distribution,
            var_eps=0,
        )

        self.z_decoder = Decoder(
            n_input=self.n_latent, n_output=n_input_regions, n_hidden=self.n_hidden
        )

        self.d_encoder = None
        if self.model_depth:
            # Decoder class to avoid variational split
            self.d_encoder = Decoder(n_input=n_input_encoder, n_output=1, n_hidden=self.n_hidden)
        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))

    def get_reconstruction_loss(self, p, d, f, x):
        """returns BCE reconstruction loss"""
        rl = torch.nn.BCELoss(reduction="none")(p * d * f, (x > 0).float()).sum(dim=-1)
        return rl

    def inference(self, x):
        """Helper function used in forward pass."""
        encoder_input = x
        # if encode_covariates is False, cat_list to init encoder is None, so
        # batch_index is not used (or categorical_input, but it's empty)
        qz_m, qz_v, z = self.z_encoder(encoder_input)
        d = self.d_encoder(encoder_input) if self.model_depth else 1

        return dict(d=d, qz_m=qz_m, qz_v=qz_v, z=z)

    def generative(self, inference_outputs, use_z_mean=False):
        """Runs the generative model."""
        z = inference_outputs["z"]
        qz_m = inference_outputs["z"]

        latent = z if not use_z_mean else qz_m
        decoder_input = latent
        p = self.z_decoder(decoder_input)

        return dict(p=p)

    def loss(self, x, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """return loss function"""
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        d = inference_outputs["d"]
        p = generative_outputs["p"]

        kld = kl_divergence(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        f = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        rl = self.get_reconstruction_loss(p, d, f, x)

        loss = (rl.sum() + kld * kl_weight).sum()

        return dict(loss=loss, rl=rl, kld=kld)


class ModTransferPEAKVI(PEAKVAE):
    """
    AE - PEAKVI Variational auto-encoder model.
    AE inputs mod1 data and outputs mod2 data
    PEAKVI VAE inputs mod2 and do PEAKVI model
    """

    def __init__(self, mod1_dim, mod2_dim, feat_dim, hidden_dim):
        super(ModTransferPEAKVI, self).__init__(mod2_dim)
        self.autoencoder = AutoEncoder(mod1_dim, mod2_dim, feat_dim, hidden_dim)
        self.peakvae = PEAKVAE(
            n_input_regions=mod2_dim,
            n_hidden=None,
            n_latent=None,
            dropout_rate=0.1,
            model_depth=True,
            region_factors=True,
            latent_distribution="normal",
        )

    def forward(self, mod1_x):
        """runs mod transfer then peakvi"""
        mod2_rec = torch.clamp(self.autoencoder(mod1_x), min=0)
        inference_outputs = self.peakvae.inference(mod2_rec)
        generative_outputs = self.peakvae.generative(inference_outputs)

        return mod2_rec, inference_outputs, generative_outputs


class PEAKVIModTransfer(PEAKVAE):
    """
    PEAKVI Variational Auto-Encoder - AE model.
    PEAKVI VAE inputs mod1 and do PEAKVI model described in
    AE inputs denoised mod1 data and outputs mod2 data
    """

    def __init__(self, mod1_dim, mod2_dim, feat_dim, hidden_dim):
        super(PEAKVIModTransfer, self).__init__(mod1_dim)
        self.peakvae = PEAKVAE(
            n_input_regions=mod1_dim,
            n_hidden=None,
            n_latent=None,
            dropout_rate=0.1,
            model_depth=True,
            region_factors=True,
            latent_distribution="normal",
        )
        self.autoencoder = AutoEncoder(mod1_dim, mod2_dim, feat_dim, hidden_dim)

    def forward(self, mod1_x):
        """runs peakvi then mod transfer"""
        inference_outputs = self.peakvae.inference(mod1_x)
        generative_outputs = self.peakvae.generative(inference_outputs)
        mod2_rec = self.autoencoder(generative_outputs["p"])

        return mod2_rec, inference_outputs, generative_outputs


if __name__ == "__main__":
    mod1_input = 235794
    mod2_input = 26717
    c_dim = 20
    bsz = 5

    # """
    x1 = torch.rand(bsz, mod1_input).cuda()
    x2 = torch.rand(bsz, mod2_input).cuda()

    vae = PEAKVAE(mod1_input).cuda()
    print(vae)
    inference_out = vae.inference(x1)
    print(inference_out)
    generative_out = vae.generative(inference_out)
    print(generative_out)

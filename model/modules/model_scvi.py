"""
The SCVI module contains simplified code from the python scvi package.
https://github.com/YosefLab/scvi-tools/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Callable, Iterable, List, Optional, NamedTuple

from modules._negative_binomial import NegativeBinomial, ZeroInflatedNegativeBinomial
from modules.model_ae import AutoEncoder

def reparameterize_gaussian(mu, var):
    """
    mu: mean from the encoder's latent space
    var: variance from the encoder's latent space
    """
    return Normal(mu, var.sqrt()).rsample()

def identity(x):
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
        var_activation: Optional[Callable] = None
        ):

        super(Encoder, self).__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
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
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent

class DecoderSCVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions into ``n_output``dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    """
    def __init__(
        self, 
        n_input: int,
        n_output: int,
        n_hidden: int = 128
    ):
        super(DecoderSCVI, self).__init__()
        self.px_decoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01),
            nn.ReLU()
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion, z: torch.Tensor, library: torch.Tensor
    ):
        """
        The forward computation for a single sample.
         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
        Parameters
        ----------
        dispersion
            * implement gene dispersion only
            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        
        return px_scale, px_r, px_rate, px_dropout

# VAE reconstruction model
class VAE(nn.Module):
    """
    Variational auto-encoder model.
    This is an implementation of the scVI model described in [Lopez18]_
    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of
        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        mod1_dim: int,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood

        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))

        self.autoencoder = AutoEncoder(
            mod1_dim, 
            out_dim=n_input, 
            feat_dim=128, 
            hidden_dim=1000
        )

        n_input_encoder = n_input
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            var_activation=var_activation,
        )

        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            var_activation=var_activation,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_hidden=n_hidden,
        )
    
    def mod2predict(self, x):
        mod2_rec = self.autoencoder(x)
        return mod2_rec

    def inference(self, x):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        encoder_input = x_
        qz_m, qz_v, z = self.z_encoder(encoder_input)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(encoder_input)
            library = library_encoded

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs

    def generative(
        self,
        inference_outputs
    ):
        """Runs the generative model."""
        
        z = inference_outputs["z"]
        library = inference_outputs["library"]

        decoder_input = z

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, decoder_input, library
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    def loss(
        self,
        x,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, qz_v), Normal(mean, scale)).sum(dim=1)
    
        kl_divergence_l = 0.0
        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return dict(
            loss=loss, reconst_loss=reconst_loss, kl_local=kl_local, kl_global=kl_global
        )

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout) -> torch.Tensor:
        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss
    
    """
    get_normalized_expression (inference stage)
    Can be refer to: 
    https://github.com/YosefLab/scvi-tools/blob/31fde3f4cf7d597cd94f1700035ce309ef0fbff7/scvi/model/base/_rnamixin.py#L41
    """

class ModTransferVAE(VAE):
    """
    AE - Variational auto-encoder model.
    AE inputs mod1 data and outputs mod2 data
    VAE inputs mod2 and do scVI model described in [Lopez18]_
    """
    def __init__(self, mod1_dim, mod2_dim, feat_dim, hidden_dim):
        super(ModTransferVAE, self).__init__(mod2_dim)
        self.autoencoder = AutoEncoder(mod1_dim, mod2_dim, feat_dim, hidden_dim)
        self.vae = VAE(
            n_input=mod2_dim,
            n_batch=0,
            n_labels=0,
            n_hidden=128,
            n_latent=10,
            dropout_rate=0.1,
            dispersion="gene",
            log_variational=True,
            gene_likelihood="zinb",
            latent_distribution="normal",
            use_observed_lib_size=True,
            var_activation=None,
        )

    def forward(self, mod1_x):
        mod2_rec = self.autoencoder(mod1_x)
        inference_outputs = self.vae.inference(mod2_rec)
        generative_outputs = self.vae.generative(inference_outputs)

        return mod2_rec, inference_outputs, generative_outputs


if __name__ == "__main__":
    mod1_input = 134
    mod2_input = 26717
    c_dim = 20
    n_hidden = 128
    n_output = 10
    bsz = 5
    
    # """
    x1 = torch.rand(bsz, mod1_input).cuda()
    x2 = torch.rand(bsz, mod2_input).cuda()
    
    vae_mod2 = VAE(mod1_input, mod2_input).cuda()
    print(vae_mod2)
    inference_outputs = vae_mod2.inference(x2)
    print(inference_outputs)

    generative_outputs = vae_mod2.generative(inference_outputs)
    print(generative_outputs)
    
    vae_mod1 = VAE(mod1_input, mod2_input).cuda()
    print(vae_mod1)
    mod2_rec = vae_mod1.mod2predict(x1)
    mod2_rec = torch.clamp(mod2_rec , min=0)
    inference_outputs = vae_mod1.inference(mod2_rec)
    print(inference_outputs)
    generative_outputs = vae_mod1.generative(inference_outputs)
    rint(generative_outputs)


# Adapted from CompVis/latent-diffusion. LDM deps replaced with local ae_modules.

import torch
import torch.nn as nn

from model.ae_modules.encoder_decoder import Encoder, Decoder
from model.ae_modules.distributions import DiagonalGaussianDistribution
from model.ae_modules.losses import LPIPSWithDiscriminator


class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, lossconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = LPIPSWithDiscriminator(**lossconfig)

        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def training_step(self, inputs, optimizer_idx, global_step):
        reconstructions, posterior = self(inputs)
        loss, log = self.loss(
            inputs,
            reconstructions,
            posterior,
            optimizer_idx,
            global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        return loss, log, reconstructions

    @torch.no_grad()
    def validation_step(self, inputs, global_step):
        reconstructions, posterior = self(inputs, sample_posterior=False)
        ae_loss, ae_log = self.loss(
            inputs, reconstructions, posterior, 0, global_step,
            last_layer=self.get_last_layer(), split="val",
        )
        disc_loss, disc_log = self.loss(
            inputs, reconstructions, posterior, 1, global_step,
            last_layer=self.get_last_layer(), split="val",
        )
        log = {**ae_log, **disc_log}
        return ae_loss, log, reconstructions
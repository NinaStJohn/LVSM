import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        kl_weight=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        logvar_init=0.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_ndf=64,
        use_actnorm=False,          # kept for yaml compat, unused
        disc_conditional=False,     # kept for yaml compat, unused
        disc_loss="hinge",
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight
        self.disc_start = disc_start

        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        # Learned log-variance for reconstruction loss
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            ndf=disc_ndf,
            n_layers=disc_num_layers,
        )

        assert disc_loss in {"hinge", "vanilla"}
        self.disc_loss_fn = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.disc_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, last_layer, split="train"):
        # Pixel-space reconstruction loss (NLL under learned variance)
        rec_loss = torch.abs(inputs - reconstructions)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.mean(nll_loss)

        kl_loss = posteriors.kl()
        kl_loss = torch.mean(kl_loss)

        if optimizer_idx == 0:
            # Generator / AE update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
            except RuntimeError:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss": loss.detach(),
                f"{split}/kl_loss": kl_loss.detach(),
                f"{split}/nll_loss": nll_loss.detach(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/g_loss": g_loss.detach(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/logvar": self.logvar.detach(),
            }
            return loss, log

        if optimizer_idx == 1:
            # Discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            disc_factor = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start)
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.detach(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log
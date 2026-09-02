"""Generative model architectures: VAE, cVAE, DCGAN, WGAN-GP, Diffusion (DDPM)."""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """VAE Encoder: 32x32 image -> latent distribution (mu, log-var)

    - Wider channels (64 -> 128 -> 256) for richer feature maps
    - GroupNorm for stable training (independent of batch size)
    - LeakyReLU to avoid dead neurons
    """

    def __init__(self, latent_dim=128, img_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    """VAE Decoder: latent sample -> 32x32 image

    - Upsample + Conv instead of ConvTranspose2d to eliminate checkerboard artifacts
    - GroupNorm + ReLU for stable training
    - Matching wider channels (256 -> 128 -> 64 -> 3)
    """

    def __init__(self, latent_dim=128, img_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, latent_dim=128, img_channels=3):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim, img_channels)
        self.decoder = VAEDecoder(latent_dim, img_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma * eps (reparameterization trick)"""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, num_samples, temperature=1.0):
        z = torch.randn(num_samples, self.latent_dim, device=next(self.parameters()).device) * temperature
        return self.decoder(z)


# ---------------------------------------------------------------------------
# cVAE
# ---------------------------------------------------------------------------

class CVAEEncoder(nn.Module):
    """Conditional VAE Encoder: 32x32 image + label -> latent (mu, logvar)"""

    def __init__(self, latent_dim=128, img_channels=3, num_classes=10, embed_dim=32):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels + embed_dim, 64, 4, stride=2, padding=1),  # 32->16
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16->8
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8->4
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x, labels):
        label_emb = self.label_embed(labels)
        label_emb = label_emb.view(label_emb.size(0), label_emb.size(1), 1, 1)
        label_emb = label_emb.expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, label_emb], dim=1)
        h = self.conv(x_cond).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    """Conditional VAE Decoder: latent + label -> 32x32 image"""

    def __init__(self, latent_dim=128, img_channels=3, num_classes=10, embed_dim=32):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            # 4->8
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            # 8->16
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            # 16->32
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_emb = self.label_embed(labels)
        z_cond = torch.cat([z, label_emb], dim=1)
        h = self.fc(z_cond).view(z_cond.size(0), 256, 4, 4)
        return self.deconv(h)


class CVAE(nn.Module):
    """Conditional Variational Autoencoder"""

    def __init__(self, latent_dim=128, img_channels=3, num_classes=10, embed_dim=32):
        super().__init__()
        self.encoder = CVAEEncoder(latent_dim, img_channels, num_classes, embed_dim)
        self.decoder = CVAEDecoder(latent_dim, img_channels, num_classes, embed_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma * eps (reparameterization trick)"""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, labels), mu, logvar

    def sample(self, num_samples, labels=None, temperature=1.0):
        """Generate new conditional samples"""
        if labels is None:
            labels = torch.randint(
                0, self.decoder.label_embed.num_embeddings, (num_samples,),
                device=next(self.parameters()).device,
            )
        z = torch.randn(num_samples, self.latent_dim, device=next(self.parameters()).device) * temperature
        return self.decoder(z, labels)


# ---------------------------------------------------------------------------
# DCGAN
# ---------------------------------------------------------------------------

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class DCDiscriminator(nn.Module):
    def __init__(self, image_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ---------------------------------------------------------------------------
# WGAN-GP
# ---------------------------------------------------------------------------

class WGANGPGenerator(nn.Module):
    """WGAN-GP Generator: noise -> 32x32 images in [-1,1]"""

    def __init__(self, latent_dim, img_channels):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(self.fc(z).view(z.size(0), 512, 4, 4))


class WGANGPCritic(nn.Module):
    """WGAN-GP Critic: CRITICAL - No BatchNorm to preserve Lipschitz constraint"""

    def __init__(self, img_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)


# ---------------------------------------------------------------------------
# Diffusion (DDPM)
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    """DDPM linear noise schedule with forward (add_noise) and reverse (reverse_step) processes.

    Based on the GaussianDiffusion formulation from Ho et al. 2020:
      forward:  x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps
      reverse:  mu_theta = (1/sqrt(alpha_t))(x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta),
                x_{t-1} = mu_theta + sigma_t * z
    """

    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, x0, t, noise):
        """Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps"""
        alpha_t = self.sqrt_alphas_cumprod[t]
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t]
        while len(alpha_t.shape) < len(x0.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        return alpha_t * x0 + sigma_t * noise

    def reverse_step(self, x, t, noise_pred):
        """Reverse process: one DDPM denoising step x_t -> x_{t-1}"""
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t])
        beta_over_sqrt = self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t]
        mean = sqrt_recip_alpha_t * (x - beta_over_sqrt * noise_pred)
        if t == 0:
            return mean
        noise = torch.randn_like(x)
        return mean + torch.sqrt(self.posterior_variance[t]) * noise

    def ddim_step(self, x_t, t, t_prev, noise_pred):
        """Deterministic DDIM step (eta=0): x_t -> x_{t_prev} (Song et al. 2020)"""
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        return torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1.0 - alpha_prev) * noise_pred


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep conditioning."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResnetBlock(nn.Module):
    """Residual block with time-embedding injection via FiLM-style additive conditioning."""

    def __init__(self, dim, time_emb_dim, out_dim=None):
        super().__init__()
        self.out_dim = out_dim or dim
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, self.out_dim))
        self.conv1 = nn.Conv2d(dim, self.out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, dim)
        self.norm2 = nn.GroupNorm(4, self.out_dim)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, time_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.mlp(time_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return self.shortcut(x) + h


class SimpleUNet(nn.Module):

    def __init__(self, img_channels=3, model_channels=128):
        super().__init__()
        C = model_channels
        time_dim = C * 4

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(C),
            nn.Linear(C, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(img_channels, C, 3, padding=1)

        self.down1_res = ResnetBlock(C, time_dim)
        self.down1_pool = nn.Conv2d(C, C, 3, stride=2, padding=1)

        self.down2_res = ResnetBlock(C, time_dim, out_dim=C * 2)
        self.down2_pool = nn.Conv2d(C * 2, C * 2, 3, stride=2, padding=1)

        self.mid_res1 = ResnetBlock(C * 2, time_dim)
        self.mid_res2 = ResnetBlock(C * 2, time_dim)

        self.up2_upsample = nn.ConvTranspose2d(C * 2, C, 4, stride=2, padding=1)
        self.up2_res = ResnetBlock(C * 3, time_dim, out_dim=C)

        self.up1_upsample = nn.ConvTranspose2d(C, C, 4, stride=2, padding=1)
        self.up1_res = ResnetBlock(C * 2, time_dim, out_dim=C)

        self.out_conv = nn.Conv2d(C, img_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t.float())

        h_init = self.init_conv(x)

        h1 = self.down1_res(h_init, t_emb)
        h1_pool = self.down1_pool(h1)

        h2 = self.down2_res(h1_pool, t_emb)
        h2_pool = self.down2_pool(h2)

        h_mid = self.mid_res1(h2_pool, t_emb)
        h_mid = self.mid_res2(h_mid, t_emb)

        h_up2 = self.up2_upsample(h_mid)
        h_up2 = self.up2_res(torch.cat([h_up2, h2], dim=1), t_emb)

        h_up1 = self.up1_upsample(h_up2)
        h_up1 = self.up1_res(torch.cat([h_up1, h1], dim=1), t_emb)

        return self.out_conv(h_up1)

"""Loss functions for VAE, cVAE, and WGAN-GP."""

import torch
import torch.nn as nn


def linear_kl_beta(epoch, beta_start=0.0, beta_end=0.5, warmup_epochs=30):
    progress = min(1.0, epoch / float(warmup_epochs))
    return beta_start + (beta_end - beta_start) * progress


def vae_loss(recon, x, mu, logvar, beta=0.5, recon_type="l1", free_bits=0.5):
    if recon_type == "l1":
        recon_loss = nn.functional.l1_loss(recon, x, reduction="mean")
    else:
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.clamp(kl, min=free_bits)
    kl_loss = torch.mean(kl)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def cvae_loss(recon, x, mu, logvar, beta=0.5, recon_type="l1", free_bits=0.5):
    """ELBO loss for cVAE with free bits, matching the VAE loss."""
    if recon_type == "l1":
        recon_loss = nn.functional.l1_loss(recon, x, reduction="mean")
    else:
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.clamp(kl, min=free_bits)
    kl_loss = torch.mean(kl)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def gradient_penalty(critic, real_images, fake_images, device, lambda_gp=10):
    batch_size = real_images.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)

    critic_score = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients_flat = gradients.view(batch_size, -1)
    grad_norm = torch.sqrt(torch.sum(gradients_flat ** 2, dim=1) + 1e-12)
    grad_penalty = lambda_gp * torch.mean((grad_norm - 1) ** 2)

    return grad_penalty

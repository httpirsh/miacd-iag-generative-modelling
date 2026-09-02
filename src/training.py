"""Training loops for VAE, cVAE, DCGAN, WGAN-GP, and Diffusion."""

import torch
import torch.nn as nn
import torch.optim as optim

from losses import cvae_loss, gradient_penalty, linear_kl_beta, vae_loss
from models import DiffusionSchedule


class EMA:
    """Exponential Moving Average of model parameters for stabler generation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters() if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def train_vae(
    model,
    train_loader,
    num_epochs,
    learning_rate,
    device,
    beta=0.5,
    beta_start=0.0,
    warmup_epochs=30,
    recon_type="l1",
    clip_grad=1.0,
):
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {"total_loss": [], "recon_loss": [], "kl_loss": [], "beta": []}

    for epoch in range(num_epochs):
        model.train()

        current_beta = linear_kl_beta(
            epoch + 1,
            beta_start=beta_start,
            beta_end=beta,
            warmup_epochs=warmup_epochs,
        )

        total_sum, recon_sum, kl_sum = 0, 0, 0

        for x, _, _ in train_loader:
            x = x.to(device)

            recon, mu, logvar = model(x)

            total_loss, recon_loss, kl_loss_val = vae_loss(
                recon, x, mu, logvar,
                beta=current_beta,
                recon_type=recon_type,
            )

            optimizer.zero_grad()
            total_loss.backward()

            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            total_sum += total_loss.item()
            recon_sum += recon_loss.item()
            kl_sum += kl_loss_val.item()

        scheduler.step()

        history["total_loss"].append(total_sum / len(train_loader))
        history["recon_loss"].append(recon_sum / len(train_loader))
        history["kl_loss"].append(kl_sum / len(train_loader))
        history["beta"].append(current_beta)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}: "
                f"Total={history['total_loss'][-1]:.4f}, "
                f"Recon={history['recon_loss'][-1]:.4f}, "
                f"KL={history['kl_loss'][-1]:.4f}, "
                f"beta={current_beta:.3f}, "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )

    return model, history


def train_cvae(
    model,
    train_loader,
    num_epochs,
    learning_rate,
    device,
    beta=0.5,
    beta_start=0.0,
    warmup_epochs=20,
    recon_type="l1",
    free_bits=0.5,
    clip_grad=1.0,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {"total_loss": [], "recon_loss": [], "kl_loss": [], "beta": []}

    for epoch in range(num_epochs):
        current_beta = linear_kl_beta(epoch + 1, beta_start=beta_start, beta_end=beta, warmup_epochs=warmup_epochs)
        model.train()
        total_sum, recon_sum, kl_sum = 0, 0, 0

        for x, labels, _ in train_loader:  # labels are required for cVAE
            x = x.to(device)
            labels = labels.to(device)

            recon, mu, logvar = model(x, labels)
            total_loss, recon_loss, kl_loss_val = cvae_loss(
                recon, x, mu, logvar, beta=current_beta, recon_type=recon_type, free_bits=free_bits
            )

            optimizer.zero_grad()
            total_loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_sum += total_loss.item()
            recon_sum += recon_loss.item()
            kl_sum += kl_loss_val.item()

        scheduler.step()
        history["total_loss"].append(total_sum / len(train_loader))
        history["recon_loss"].append(recon_sum / len(train_loader))
        history["kl_loss"].append(kl_sum / len(train_loader))
        history["beta"].append(current_beta)

        if (epoch + 1) % 10 == 0:
            print(
                f"cVAE Epoch {epoch+1:3d}: Total={history['total_loss'][-1]:.4f}, "
                f"Recon={history['recon_loss'][-1]:.4f}, KL={history['kl_loss'][-1]:.4f}, "
                f"beta={current_beta:.3f}, LR={scheduler.get_last_lr()[0]:.2e}"
            )

    return model, history


def train_dcgan(gen, disc, train_loader, num_epochs, learning_rate, latent_dim, device):
    gen, disc = gen.to(device), disc.to(device)

    opt_g = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # BCEWithLogitsLoss combines sigmoid + binary cross-entropy
    criterion = nn.BCEWithLogitsLoss()
    history = {"g_loss": [], "d_loss": []}

    for epoch in range(num_epochs):
        g_sum, d_sum = 0, 0
        for x, _, _ in train_loader:
            x = x.to(device)
            bs = x.size(0)

            # Train Discriminator
            opt_d.zero_grad()
            real_out = disc(x)  # [batch]

            # Label smoothing: real=0.9, fake=0.1
            real_labels = torch.ones(bs, device=device) * 0.9
            fake_labels = torch.zeros(bs, device=device) + 0.1

            d_real = criterion(real_out, real_labels)

            z = torch.randn(bs, latent_dim, device=device)
            fake = gen(z)
            d_fake = criterion(disc(fake.detach()), fake_labels)  # detach to avoid backprop to generator

            d_loss = d_real + d_fake
            d_loss.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            fake = gen(z)
            g_loss = criterion(disc(fake), torch.ones(bs, device=device))  # generator wants discriminator fooled
            g_loss.backward()
            opt_g.step()

            g_sum += g_loss.item()
            d_sum += d_loss.item()

        history["g_loss"].append(g_sum / len(train_loader))
        history["d_loss"].append(d_sum / len(train_loader))

        if (epoch + 1) % 10 == 0:
            print(f"DCGAN Epoch {epoch+1}: G={history['g_loss'][-1]:.4f}, D={history['d_loss'][-1]:.4f}")

    return gen, disc, history


def train_wgan(gen, critic, train_loader, num_epochs, latent_dim, lambda_gp, n_critic, lr_gen, lr_critic, device):
    """Train WGAN-GP, tuned for brighter images and better FID/KID."""
    gen, critic = gen.to(device), critic.to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr_critic, betas=(0.0, 0.9))

    history = {"g_loss": [], "c_loss": []}

    for epoch in range(num_epochs):
        g_sum, c_sum = 0, 0
        for x, _, _ in train_loader:
            x = x.to(device)
            bs = x.size(0)

            # Train Critic (multiple steps)
            for _ in range(n_critic):
                opt_c.zero_grad()
                z = torch.randn(bs, latent_dim, device=device)
                fake = gen(z).detach()

                # Wasserstein loss
                real_score = critic(x)
                fake_score = critic(fake)
                c_loss = -torch.mean(real_score) + torch.mean(fake_score)

                # Gradient penalty
                gp = gradient_penalty(critic, x, fake, device, lambda_gp)
                total_c_loss = c_loss + gp

                total_c_loss.backward()
                opt_c.step()
                c_sum += total_c_loss.item()

            # Train Generator
            opt_g.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            fake = gen(z)
            g_loss = -torch.mean(critic(fake))
            g_loss.backward()
            opt_g.step()
            g_sum += g_loss.item()

        history["g_loss"].append(g_sum / len(train_loader))
        history["c_loss"].append(c_sum / (len(train_loader) * n_critic))
        if (epoch + 1) % 10 == 0:
            print(f"WGAN-GP Epoch {epoch+1}: G={history['g_loss'][-1]:.4f}, C={history['c_loss'][-1]:.4f}")

    return gen, critic, history


def train_diffusion(model, train_loader, num_epochs, learning_rate, num_steps, device):
    """Train Diffusion model on dataloader."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine annealing decays LR smoothly to ~0 by the end of training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    schedule = DiffusionSchedule(num_steps=num_steps).to(device)
    ema = EMA(model, decay=0.999)
    history = {"loss": []}

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        for x, _, _ in train_loader:
            x = x.to(device)
            # Data already in [-1,1] from T.Normalize in transform

            t = torch.randint(0, num_steps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = schedule.add_noise(x, t, noise)
            noise_pred = model(x_t, t)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            loss_sum += loss.item()

        scheduler.step()
        history["loss"].append(loss_sum / len(train_loader))
        if (epoch + 1) % 10 == 0:
            print(f"Diffusion Epoch {epoch+1}: Loss={history['loss'][-1]:.4f}")

    # Replace model weights with EMA weights for generation
    ema.apply_shadow(model)
    return model, history

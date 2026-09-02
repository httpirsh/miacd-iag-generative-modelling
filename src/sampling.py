"""Sample generation from trained generative models."""

import torch


def generate_vae_samples(model, num_samples=5000, latent_dim=None, batch_size=128, temperature=0.8):
    """Generate samples from trained VAE in [-1,1] using temperature scaling."""
    model.eval()
    if latent_dim is None:
        latent_dim = model.latent_dim
    samples = []
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            batch_size_actual = min(batch_size, num_samples - start)
            sample = model.sample(batch_size_actual, temperature=temperature)
            samples.append(sample.cpu())
    return torch.cat(samples, dim=0)[:num_samples]


def generate_cvae_samples(
    model, device, num_samples=5000, latent_dim=None, batch_size=128,
    num_classes=10, class_labels=None, temperature=0.8, debug=False,
):
    """Generate samples from trained cVAE in [-1,1]."""
    model.eval()
    if latent_dim is None:
        latent_dim = model.latent_dim
    samples = []

    if class_labels is None:
        repeats = (num_samples + num_classes - 1) // num_classes
        class_labels = torch.arange(num_classes).repeat(repeats)[:num_samples]
    elif isinstance(class_labels, int):
        class_labels = torch.full((num_samples,), class_labels, dtype=torch.long)
    else:
        class_labels = torch.as_tensor(class_labels, dtype=torch.long)[:num_samples]

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_labels = class_labels[start:end].to(device)
            z = torch.randn(end - start, latent_dim, device=device) * temperature
            batch_samples = model.decoder(z, batch_labels)
            if debug:
                print(f"[DEBUG] Batch {start}-{end} range: min={batch_samples.min().item():.4f}, max={batch_samples.max().item():.4f}")
            samples.append(batch_samples.cpu())

    return torch.cat(samples, dim=0)[:num_samples]


def generate_gan_samples(generator, device, num_samples=5000, latent_dim=128, batch_size=128):
    """Generate samples from trained GAN generator in [-1,1]."""
    generator.eval()
    samples = []
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            batch_size_actual = min(batch_size, num_samples - start)
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            sample = generator(z)
            samples.append(sample.cpu())
    return torch.cat(samples, dim=0)[:num_samples]


def generate_diffusion_samples(model, schedule, device, num_samples=5000, batch_size=128, ddim_steps=50):
    """Generate samples from a trained Diffusion model in [-1,1] (ready for InceptionFeatureExtractor)."""
    model.eval()
    T = schedule.num_steps
    step_ratio = max(1, T // ddim_steps)
    inf_steps = list(range(T - 1, -1, -step_ratio))[:ddim_steps]
    samples = []
    with torch.no_grad():
        for _ in range(0, num_samples, batch_size):
            batch_size_actual = min(batch_size, num_samples - len(samples))
            x = torch.randn(batch_size_actual, 3, 32, 32, device=device)

            for i, t in enumerate(inf_steps):
                t_prev = inf_steps[i + 1] if i + 1 < len(inf_steps) else -1
                t_tensor = torch.full((batch_size_actual,), t, device=device, dtype=torch.long)
                noise_pred = model(x, t_tensor)
                x = schedule.ddim_step(x, t, t_prev, noise_pred)

            samples.append(x.cpu().clamp(-1, 1))
    return torch.cat(samples, dim=0)[:num_samples]


def get_real_samples(data_loader, device, num_samples=5000):
    """Collect real samples from dataloader in [-1,1]"""
    real_samples = []
    with torch.no_grad():
        for x, _, _ in data_loader:
            real_samples.append(x)
            if sum(s.size(0) for s in real_samples) >= num_samples:
                break
    samples = torch.cat(real_samples, dim=0)[:num_samples]
    return samples.to(device)

"""Visualization helpers for datasets, generated samples, and training curves."""

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def to_display_range(x):
    """Convert image tensor from [-1,1] to [0,1] for visualization/saving only."""
    return ((x + 1.0) / 2.0).clamp(0, 1)


def show_batch_grid(loader, class_names, n_images=36, nrow=6, title="Sample Grid"):
    x, y, idx = next(iter(loader))
    x = x[:n_images]
    y = y[:n_images]

    x_vis = to_display_range(x)
    grid = make_grid(x_vis, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_img)
    plt.axis("off")
    plt.title(title)
    plt.show()
    labels_str = [class_names[int(v)] for v in y]
    print("Labels:", labels_str)


def show_generated_samples(samples, n_images=36, nrow=6, title="Generated Samples"):
    samples_vis = to_display_range(samples[:n_images])
    grid = make_grid(samples_vis, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def show_cvae_class_grid(model, num_classes=10, samples_per_class=8, latent_dim=128, temperature=0.8):
    """Generate and display a grid of cVAE samples, organized by class label."""
    model.eval()
    all_images = []

    with torch.no_grad():
        for label in range(num_classes):
            labels = torch.full((samples_per_class,), label, device=next(model.parameters()).device)
            z = torch.randn(samples_per_class, latent_dim, device=next(model.parameters()).device) * temperature
            samples = model.decoder(z, labels)  # Output in [-1,1]
            all_images.append(samples.cpu())

    all_images = torch.cat(all_images, dim=0)  # (num_classes*samples_per_class, C, H, W)
    all_images = to_display_range(all_images)

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class, num_classes))
    for i in range(num_classes):
        for j in range(samples_per_class):
            img = all_images[i * samples_per_class + j].permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_history(history_dict, model_name="Model"):
    fig, axes = plt.subplots(1, len(history_dict), figsize=(5 * len(history_dict), 4))
    if len(history_dict) == 1:
        axes = [axes]

    for (key, losses), ax in zip(history_dict.items(), axes):
        ax.plot(losses, linewidth=2)
        ax.set_title(f"{model_name} - {key}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# ArtBench-10 Generative Modelling

Comparing generative models — **VAE**, **cVAE**, **DCGAN**, **WGAN-GP**, and a **Diffusion model** — trained from scratch on [ArtBench-10](https://www.kaggle.com/datasets/alexanderliao/artbench10), a dataset of 60,000 32×32 artwork images spanning 10 artistic styles. Model quality is compared using FID and KID scores against real samples.

This was built as a project for the *Generative Modelling* course of the Master's in Data Science (MIACD).

## Approach

1. **Phase 1 — Development (20% subset):** all five models are trained and evaluated on a fixed 20% training split (`data/training_20_percent.csv`) to quickly compare architectures via FID/KID.
2. **Phase 2 — Final evaluation (100% dataset):** the best-performing model from Phase 1 is retrained on the full dataset across multiple random seeds to obtain robust FID/KID statistics (mean ± std).

### Phase 1 results (20% subset, single seed)

| Model     | FID     | KID (mean ± std)     |
|-----------|---------|-----------------------|
| VAE       | 154.01  | 0.0643 ± 0.0036      |
| cVAE      | 165.80  | 0.0687 ± 0.0031      |
| DCGAN     | 80.58   | 0.0308 ± 0.0033      |
| WGAN-GP   | 119.18  | 0.0486 ± 0.0037      |
| **Diffusion** | **40.34** | **0.0134 ± 0.0018** |

Lower is better for both metrics. The Diffusion model was selected as the best performer and retrained on the full dataset in Phase 2.

## Project structure

```
.
├── notebooks/
│   └── ArtBench10_Student.ipynb    # Orchestrates data loading, training, and evaluation using src/
├── src/
│   ├── artbench_local_dataset.py   # Loads ArtBench-10 from local Kaggle-format files into a Hugging Face DatasetDict
│   ├── data.py                     # PyTorch Dataset/transform/dataloader helpers
│   ├── models.py                   # VAE, cVAE, DCGAN, WGAN-GP, and Diffusion (DDPM) architectures
│   ├── losses.py                   # VAE/cVAE ELBO loss and WGAN-GP gradient penalty
│   ├── training.py                 # Training loops for each model
│   ├── sampling.py                 # Sample generation from trained models
│   ├── metrics.py                  # FID/KID evaluation via Inception-v3 features
│   ├── viz.py                      # Plotting and sample-grid visualization
│   └── utils.py                    # Seeding and results-reporting helpers
├── data/
│   ├── artbench-10-python/         # ArtBench-10 in Python pickle format (used by the notebook)
│   ├── artbench-10-binary/         # ArtBench-10 in binary batch format
│   ├── ArtBench-10.csv             # Dataset metadata and labels
│   └── training_20_percent.csv     # Phase 1 training subset definition
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is already included under `data/`, so no download step is needed.

## Running

```bash
jupyter notebook notebooks/ArtBench10_Student.ipynb
```

Run the cells in order: dataset loading → dataloaders → model definitions → Phase 1 training/evaluation → Phase 2 final training on the full dataset.

## Models

- **VAE** — encoder/decoder with a reparameterized Gaussian latent space; reconstruction + KL loss.
- **cVAE** — VAE conditioned on class labels for controlled, class-consistent generation.
- **DCGAN** — convolutional generator/discriminator trained adversarially.
- **WGAN-GP** — Wasserstein GAN with a gradient penalty for more stable adversarial training.
- **Diffusion** — a UNet trained to reverse a fixed Gaussian noising process (DDPM-style).

## Authors

- Íris Sousa
- Bernardo Pedro

## References

- [ArtBench-10 dataset](https://www.kaggle.com/datasets/alexanderliao/artbench10)
- [FID: GANs Trained by a Two Time-Scale Update Rule](https://arxiv.org/abs/1706.08500)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

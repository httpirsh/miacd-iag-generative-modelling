"""FID and KID evaluation metrics via Inception-v3 features."""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class InceptionFeatureExtractor:
    """Extract features from pre-trained Inception-v3.

    Contract: input images must be in [-1,1]. This class converts to [0,1] only
    internally, right before ImageNet normalization required by Inception.
    """

    def __init__(self, device="cpu"):
        # aux_logits must be True for the pre-trained weights
        inception = models.inception_v3(weights="DEFAULT", aux_logits=True)

        # Extract features from the last average pooling layer, before the classification head
        self.feature_extractor = create_feature_extractor(
            inception, return_nodes={"avgpool": "features"}
        )

        self.device = device
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def extract_models(self, images):
        """Extract features for a batch of images in [-1,1]."""
        with torch.no_grad():
            images = images.to(self.device).float()

            # Resize to 299x299 (required by Inception-v3)
            images_resized = nn.functional.interpolate(
                images, size=(299, 299), mode="bilinear", align_corners=False
            )

            # Convert to [0,1] for FID/KID calculation (Inception expects this range)
            images_01 = ((images_resized + 1.0) / 2.0).clamp(0, 1)

            # Normalize using ImageNet statistics
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            images_normalized = (images_01 - mean) / std

            feature_dict = self.feature_extractor(images_normalized)
            features = feature_dict["features"]  # Shape: (batch, 2048, 1, 1)

            return features.view(features.size(0), -1)


def extract_features_from_tensor(samples_tensor, feature_extractor, device, batch_size=128):
    """Extract Inception features from tensor samples (expects samples in [-1,1])."""
    all_features = []

    with torch.no_grad():
        for i in range(0, len(samples_tensor), batch_size):
            batch = samples_tensor[i:i + batch_size].to(device)
            features = feature_extractor.extract_models(batch)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def calculate_fid(real_features, generated_features):
    """Calculate Frechet Inception Distance (FID) between real and generated features."""
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_gen
    diff_squared = np.dot(diff, diff)

    covmean = sqrtm(sigma_real @ sigma_gen)

    # Numerical stability: handle complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff_squared + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid_score


def calculate_kid(real_features, generated_features, n_subsets=50, subset_size=100):
    """Calculate Kernel Inception Distance with random subsets for variance estimation."""
    n_real = len(real_features)
    n_gen = len(generated_features)

    kid_values = []
    for _ in range(n_subsets):
        real_idx = np.random.choice(n_real, subset_size, replace=False)
        gen_idx = np.random.choice(n_gen, subset_size, replace=False)

        real_subset = real_features[real_idx]
        gen_subset = generated_features[gen_idx]

        K_rr = rbf_kernel(real_subset)
        K_gg = rbf_kernel(gen_subset)
        K_rg = rbf_kernel(real_subset, gen_subset)

        kid = np.mean(K_rr) + np.mean(K_gg) - 2 * np.mean(K_rg)
        kid_values.append(max(0, kid))  # KID should be non-negative

    return np.mean(kid_values), np.std(kid_values)

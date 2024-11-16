"""
Image preprocessing utility for training and testing vision models.

This module provides a function to apply preprocessing transformations to image data, 
using PyTorch's `torchvision.transforms`. The transformations vary depending on whether 
the model is being trained or evaluated.

Functions:
    - get_preprocessing: Returns a composed set of transformations for image preprocessing
      based on whether the model is in training or evaluation mode.

"""

import torchvision.transforms as T
from .config_visual import VisualConfig


def get_preprocessing(config: VisualConfig, is_training=True):
    """
    Returns the image transformations for training or evaluation based on the config.

    Depending on the `is_training` flag, different sets of transformations are applied to 
    the images to augment the dataset for training or simply resize for evaluation.

    Args:
        config (VisualConfig): Configuration object containing the preprocessing settings.
        is_training (bool, optional): Flag indicating whether the transformations are for 
                                      training or evaluation. Default is True.

    Returns:
        torchvision.transforms.Compose: A composed sequence of image transformation functions.

    """

    transform = []

    if is_training:
        transform.append(T.RandomRotation(15))  # Random rotation for data augmentation
        transform.append(
            T.RandomResizedCrop(size=config.train_input_size, scale=(0.9, 1))  # Random crop
        )

        transform.append(
            T.ColorJitter(
                config.aug_color_jitter_b,
                config.aug_color_jitter_c,
                config.aug_color_jitter_s,
                0.0,
            )  # Color jitter for brightness, contrast, and saturation
        )
        transform.append(T.RandomHorizontalFlip())  # Horizontal flip for augmentation
    else:
        transform.append(T.Resize(config.test_input_size))  # Resize for evaluation

    # Common transformations
    transform.append(T.ToTensor())  # Convert image to tensor
    transform.append(T.Normalize(config.norm_mean, config.norm_std))  # Normalize image

    return T.Compose(transform)

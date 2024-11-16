"""
Configuration module for visual classification projects.

This module defines a `VisualConfig` class that holds
the configuration settings for visual classification tasks,
including model parameters, preprocessing settings,
dataset paths, and optimization configurations.
It also provides methods to save and load the configuration 
from JSON files.

The `VisualConfig` class is used to manage hyperparameters and other settings for the visual 
classification pipeline, ensuring that configurations are easily shareable and reproducible.

Methods:
    - to_json(path: str): Saves the current configuration as a JSON file.
    - from_json(path: str): Loads a config. from a JSON file and returns a `VisualConfig` instance.

Attributes:
    - project_name (str): The name of the project.
    - random_state (int): Random seed for reproducibility.
    - device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
    - seed (int): Seed for random number generation.
    - num_classes (int): The number of classes in the classification task.
    - model_name (str): Name of the visual model (e.g., 'resnet18').
    - pretrained (bool): Whether to use a pretrained model.
    - train_input_size (Tuple[int, int]): Input size for training images.
    - test_input_size (Tuple[int, int]): Input size for test images.
    - aug_color_jitter_b (float): Strength of the brightness jitter for data augmentation.
    - aug_color_jitter_c (float): Strength of the contrast jitter for data augmentation.
    - aug_color_jitter_s (float): Strength of the saturation jitter for data augmentation.
    - norm_mean (Tuple[float, float, float]): Mean values for normalization.
    - norm_std (Tuple[float, float, float]): Standard deviation values for normalization.
    - fold (int): The fold to use for cross-validation; -1 means no evaluation.
    - csv_train_file (str): Path to the training dataset CSV file.
    - csv_test_file (str): Path to the test dataset CSV file.
    - csv_split_file (str): Path to the CSV file containing cross-validation fold splits.
    - root_train_images (str): Path to the directory containing training images.
    - root_test_images (str): Path to the directory containing test images.
    - num_epochs (int): Number of training epochs.
    - batch_size (int): The size of the training batches.
    - test_batch_size (int): The size of the test batches.
    - num_workers (int): Number of workers for data loading.
    - lr (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay for regularization.
"""

from dataclasses import dataclass
from typing import Tuple
import json


@dataclass
class VisualConfig:
    """
    Configuration class for visual classification projects.

    This class stores the configuration settings for the visual classification pipeline, 
    including model, dataset, preprocessing, and optimization configurations. It provides 
    methods to save and load configurations from JSON files, making it easy to manage and 
    share the settings.

    Attributes:
        project_name (str): The name of the project.
        random_state (int): Random seed for reproducibility.
        device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
        seed (int): Seed for random number generation.
        num_classes (int): The number of classes in the classification task.
        model_name (str): Name of the visual model (e.g., 'resnet18').
        pretrained (bool): Whether to use a pretrained model.
        train_input_size (Tuple[int, int]): Input size for training images.
        test_input_size (Tuple[int, int]): Input size for test images.
        aug_color_jitter_b (float): Strength of the brightness jitter for data augmentation.
        aug_color_jitter_c (float): Strength of the contrast jitter for data augmentation.
        aug_color_jitter_s (float): Strength of the saturation jitter for data augmentation.
        norm_mean (Tuple[float, float, float]): Mean values for normalization.
        norm_std (Tuple[float, float, float]): Standard deviation values for normalization.
        fold (int): The fold to use for cross-validation; -1 means no evaluation.
        csv_train_file (str): Path to the training dataset CSV file.
        csv_test_file (str): Path to the test dataset CSV file.
        csv_split_file (str): Path to the CSV file containing cross-validation fold splits.
        root_train_images (str): Path to the directory containing training images.
        root_test_images (str): Path to the directory containing test images.
        num_epochs (int): Number of training epochs.
        batch_size (int): The size of the training batches.
        test_batch_size (int): The size of the test batches.
        num_workers (int): Number of workers for data loading.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.

    Methods:
        to_json(path: str): Saves the configuration to a JSON file.
        from_json(path: str): Loads the configuration from a JSON file and returns a `VisualConfig`.
    """

    def to_json(self, path: str):
        """
        Save the current configuration to a JSON file.

        Args:
            path (str): The path to save the configuration file.

        The method serializes the current object's attributes to a JSON file,
        using the `__dict__` method to extract the instance attributes as a dictionary.
        """
        with open(path, "w", encoding="UTF-8") as fp:
            json.dump(self.__dict__, fp, indent=2)

    @classmethod
    def from_json(cls, path: str):
        """
        Load a configuration from a JSON file.

        Args:
            path (str): The path of the configuration JSON file to load.

        Returns:
            VisualConfig: An instance of the `VisualConfig` class populated with the values 
            from the loaded JSON file.

        The method reads a JSON file and initializes a `VisualConfig'.
        """
        with open(path, "r", encoding="UTF-8") as fp:
            json_obj = json.load(fp)

        return cls(**json_obj)

    # General configuration
    project_name: str = "project_visual"
    random_state: int = 42
    device: str = "cuda"
    seed: int = 42

    # Model configuration
    num_classes: int = 4
    model_name: str = "resnet18"
    pretrained: bool = True

    # Preprocessing configuration
    train_input_size: Tuple[int, int] = (224, 224)
    test_input_size: Tuple[int, int] = (224, 224)

    aug_color_jitter_b: float = 0.1
    aug_color_jitter_c: float = 0.1
    aug_color_jitter_s: float = 0.1

    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Dataset configuration
    fold: int = 0
    csv_train_file: str = "dataset/train.csv"
    csv_test_file: str = "dataset/test.csv"
    csv_split_file: str = "dataset/fold.csv"
    root_train_images: str = "dataset/images/train"
    root_test_images: str = "dataset/images/test"

    # Optimizer configuration
    num_epochs: int = 15
    batch_size: int = 32
    test_batch_size: int = 64
    num_workers: int = 0

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4

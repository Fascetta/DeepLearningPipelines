"""
Configuration module for text classification projects.

This module defines a `TextConfig` class that encapsulates all
the configuration settings required for a text classification pipeline.
It includes parameters for the model, dataset, training, and optimization, 
as well as methods to save and load configurations in JSON format.

The `TextConfig` class is designed to simplify the management of hyperparameters and other settings, 
making it easy to store, share, and reuse configurations across different runs and experiments.

Methods:
    - to_json(path: str): Saves the current configuration as a JSON file.
    - from_json(path: str): Loads a config. from a JSON file and returns a `TextConfig` instance.

Attributes:
    - project_name (str): The name of the project.
    - random_state (int): Random seed for reproducibility.
    - device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
    - seed (int): Seed for random number generation.
    - num_classes (int): The number of classes in the classification task.
    - model_name (str): Name of the transformer model (e.g., 'roberta-base').
    - pretrained (bool): Whether to use a pretrained model.
    - fold (int): The fold to use for cross-validation; -1 means no evaluation.
    - csv_train_file (str): Path to the training dataset CSV file.
    - csv_test_file (str): Path to the test dataset CSV file.
    - csv_split_file (str): Path to the CSV file containing cross-validation fold splits.
    - batched (bool): Whether to use batched processing for the dataset.
    - num_epochs (int): Number of training epochs.
    - batch_size (int): The size of the training batches.
    - test_batch_size (int): The size of the test batches.
    - num_workers (int): Number of workers for data loading.
    - lr (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay for regularization.
"""

from dataclasses import dataclass
import json


@dataclass
class TextConfig:
    """
    Configuration class for text classification projects.

    This class holds the configuration settings for the text classification pipeline, 
    including model, dataset, and training parameters. It also provides methods for 
    saving and loading configurations from JSON files.

    Attributes:
        project_name (str): The name of the project.
        random_state (int): Random seed for reproducibility.
        device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
        seed (int): Seed for random number generation.
        num_classes (int): The number of classes in the classification task.
        model_name (str): Name of the transformer model (e.g., 'roberta-base').
        pretrained (bool): Whether to use a pretrained model.
        fold (int): The fold to use for cross-validation; -1 means no evaluation.
        csv_train_file (str): Path to the training dataset CSV file.
        csv_test_file (str): Path to the test dataset CSV file.
        csv_split_file (str): Path to the CSV file containing cross-validation fold splits.
        batched (bool): Whether to use batched processing for the dataset.
        num_epochs (int): Number of training epochs.
        batch_size (int): The size of the training batches.
        test_batch_size (int): The size of the test batches.
        num_workers (int): Number of workers for data loading.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.

    Methods:
        to_json(path: str): Saves the configuration to a JSON file at the specified path.
        from_json(path: str): Loads the configuration from a JSON file at the specified path.
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
            TextConfig: An instance of the `TextConfig` class populated with the values 
            from the loaded JSON file.

        The method reads a JSON file and initializes a `TextConfig`.
        """
        with open(path, "r", encoding="UTF-8") as fp:
            json_obj = json.load(fp)

        return cls(**json_obj)

    # General configuration
    project_name: str = "project_text"
    random_state: int = 42
    device: str = "cuda"
    seed: int = 42

    # Model configuration
    num_classes: int = 4
    model_name: str = "roberta-base"
    pretrained: bool = True

    # Dataset configuration
    fold: int = 0
    csv_train_file: str = "dataset/train.csv"
    csv_test_file: str = "dataset/test.csv"
    csv_split_file: str = "dataset/fold.csv"
    batched: bool = True

    # Optimizer configuration
    num_epochs: int = 3
    batch_size: int = 16
    test_batch_size: int = 16
    num_workers: int = 0

    lr: float = 2e-5
    weight_decay: float = 1e-2

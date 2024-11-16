"""
Utility functions for managing reproducibility, computing running means,
and generating output folder names.

This module provides functions to ensure deterministic behavior during training,
maintain running statistics,
and generate timestamped folder names for organizing output files.

Functions:
    - seed_everything: Sets the random seed for various libraries to ensure reproducibility.
    - RunningMean: A class to compute the running mean of a stream of values.
    - get_output_folder: Generates a folder name based on the project name and the current time.
"""

import random
from datetime import datetime
import torch
import numpy as np


def seed_everything(seed):
    """
    Sets the random seed for various libraries to ensure reproducibility of results.

    Args:
        seed (int): The seed value to be used for random number generation across libraries.

    This function sets the seed for:
        - torch (CPU and CUDA)
        - numpy
        - random (Python's built-in random module)
        - torch.backends.cudnn for deterministic behavior
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


class RunningMean:
    """
    A class to compute the running mean of a stream of values iteratively.

    This class maintains a running mean of the values it receives
    and updates the mean with each new value.

    Attributes:
        name (str): An optional name for the running mean object.
        mean (float): The current mean value.
        n (int): The number of values processed so far.

    Methods:
        restart: Resets the running mean and count to their initial state.
        update: Updates the running mean with a new value.
        __str__: Returns a string representation of the current mean.
    """

    def __init__(self, name: str = ""):
        """
        Initializes the RunningMean object with an optional name.

        Args:
            name (str, optional): The name to identify this RunningMean object.
        """
        self.name = name
        self.mean = 0
        self.restart()

    def restart(self):
        """
        Resets the running mean and the count of values processed to 0.
        """
        self.mean = 0
        self.n = 0

    def update(self, value):
        """
        Updates the running mean with a new value.

        Args:
            value (float): The new value to be added to the running mean.
        """
        self.mean = self.mean + (value - self.mean) / (self.n + 1)
        self.n += 1

    def __str__(self):
        """
        Returns the string representation of the current mean.

        Returns:
            str: The string representation of the running mean.
        """
        return f"{self.mean}"


def get_output_folder(project_name):
    """
    Generates a folder name for storing output files,
    based on the project name and the current timestamp.

    Args:
        project_name (str): The name of the project.

    Returns:
        str: The generated folder name, including the timestamp.
    """
    return project_name + "_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

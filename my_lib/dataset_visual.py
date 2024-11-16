"""
Custom Dataset class for loading image data from a directory and associated labels from a DataFrame.

This module defines a `DFDataset` class that extends the PyTorch `Dataset` class. It is designed to 
load images from a given directory (`root`) based on file names specified in a DataFrame (`df`), 
optionally applies transformations to the images, and returns images along with their corresponding 
labels if available.

Classes:
    - DFDataset: A custom dataset class that loads image data and associated labels (if available).

"""

import os
from torch.utils.data import Dataset
import PIL.Image


class DFDataset(Dataset):
    """
    A custom dataset class to load images and labels from a DataFrame.

    Args:
        root (str): The root directory where the images are stored.
        df (pd.DataFrame): A DataFrame containing image filenames and optionally labels.
        transform (callable, optional): A function/transform to apply to the image data.

    Attributes:
        images (list): A list of file paths to the images.
        target (ndarray or None): An array of labels, or None if no labels are present.
        transform (callable): A function/transform to apply to the images during loading.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image (and label if available) at the specified index, 
                          with any specified transformations applied.

    """

    def __init__(self, root, df, transform=None):
        """
        Initializes the dataset by setting up the paths to images and optionally labels.

        Args:
            root (str): The directory containing the images.
            df (pd.DataFrame): DataFrame containing image file names and labels (if available).
            transform (callable, optional): A transform function to apply to each image.
        """
        super().__init__()
        self.images = [
            os.path.join(root, str(row[1]["image_name"])) for row in df.iterrows()
        ]

        if "target" in df.columns:
            self.target = df["target"].values
        else:
            self.target = None

        self.transform = transform

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image and its label (if available) at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and the corresponding label (if available).
        """
        img = PIL.Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)

        if self.target is None:
            return img

        return img, self.target[idx]

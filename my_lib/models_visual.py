"""
A visual model for image classification using a pretrained backbone from the Timm library.

This model uses a backbone from the Timm library as the feature extractor (encoder)
and adds a fully connected layer as the classification head.
The model is customizable with different backbone models
and can be used for various image classification tasks by specifying the desired
model architecture and number of output classes.

Args:
    model_name (str): The name of the model architecture to use.
    num_classes (int): The number of output classes for the classification task.
    pretrained (bool): Whether to use pretrained weights for the model. Default is True.

Attributes:
    encoder (nn.Module): The backbone feature extractor, using a pretrained model from Timm.
    embedding_size (int): The size of the feature map output by the encoder.
    head (nn.Module): The fully connected classification head.
"""

import timm
import torch
from torch import nn


class VisualModelTimm(nn.Module):
    """
    A visual model for image classification using a pretrained backbone from the Timm library.

    This model leverages a popular architecture from the Timm library (e.g., ResNet, EfficientNet) 
    as the encoder (feature extractor) and adds a fully connected head for classification. The model 
    can be used as a baseline for various vision tasks by simply specifying 
    the backbone model and the number of output classes.

    Args:
        model_name (str): The name of the model architecture (e.g., 'resnet34', 'efficientnet_b0').
        num_classes (int): The number of output classes for the classification task.
        pretrained (bool): Whether to load pretrained weights. Default is True.

    Attributes:
        encoder (nn.Module): Backbone feature extractor using a pretrained model.
        embedding_size (int): The size of the output feature map from the encoder.
        head (nn.Module): Fully connected layer for classification.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Initialize the model by creating the encoder from Timm and adding the classification head.

        Args:
            model_name (str): Name of the pretrained model to be used for the backbone.
            num_classes (int): Number of output classes for the final classification task.
            pretrained (bool): Whether to use pretrained weights for the backbone model.
        """
        super().__init__()

        # Validate model name
        if model_name not in timm.list_models():
            raise ValueError(f"Model name {model_name} not found in Timm model zoo.")

        # Create the model backbone (encoder)
        self.encoder = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained
        )

        # Retrieve the input size for the model to calculate the embedding size
        config = timm.get_pretrained_cfg(model_name=model_name, allow_unregistered=True)
        config = config.to_dict()

        # Use a dummy tensor to get the output shape (embedding size) of the backbone
        with torch.no_grad():
            emb = self.encoder(torch.rand(1, *config["input_size"]))
            self.embedding_size = emb.shape[1]

        # Add a fully connected layer for classification
        self.head = nn.Linear(self.embedding_size, num_classes)

    def encode(self, x):
        """
        Forward pass through the backbone encoder (without the classification head).

        This method extracts features from the input images using the encoder part of the model. 
        It does not pass through the final classification head.

        Args:
            x (Tensor): Input tensor (e.g., a batch of images).

        Returns:
            Tensor: The feature representation of the input (before the classification head).
        """
        return self.encoder(x)

    def forward(self, x):
        """
        Forward pass through the full model (encoder + classification head).

        This method first extracts features using the encoder and then passes those features through 
        the final fully connected head to produce the classification output.

        Args:
            x (Tensor): Input tensor (e.g., a batch of images).

        Returns:
            Tensor: The predicted output for the classification task.
        """
        x = self.encode(x)  # Extract features using the encoder
        return self.head(x)  # Pass the features through the classification head

"""
A text classification model using a transformer-based architecture from the HF Transformers library.

This model uses a transformer encoder from the Hugging Face model hub as the backbone,
followed by a fully connected layer for classification.
The model is customizable by specifying the desired transformer 
model architecture and the number of output classes.

Args:
    model_name (str): The name of the transformer model to use (e.g., 'bert-base-uncased').
    num_classes (int): The number of output classes for the classification task.
    pretrained (bool): Whether to load pretrained weights for the transformer model.

Attributes:
    encoder (PreTrainedModel): The transformer encoder from Hugging Face's Transformers library.
    head (nn.Module): The fully connected layer used for classification.
"""

from transformers import AutoModel
import torch
from torch import nn


class NLPClassificationModel(nn.Module):
    """
    A transformer-based text classification model using a pretrained model from the HF model hub.

    This model utilizes a transformer encoder for feature extraction and adds a 
    fully connected layer as a classification head.
    The model can be fine-tuned for any text classification task 
    by specifying the transformer model and the number of output classes.

    Args:
        model_name (str): The name of the transformer model architecture.
        num_classes (int): The number of output classes for the classification task.
        pretrained (bool): Whether to use pretrained weights for the transformer model.

    Attributes:
        encoder (PreTrainedModel): The transformer encoder from Hugging Face's Transformers library.
        head (nn.Module): A fully connected layer for classifying the features into `num_classes`.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True) -> None:
        """
        Initialize the model by loading an encoder from HF and adding a classification head.

        Args:
            model_name (str): The name of the pretrained transformer model.
            num_classes (int): The number of output classes for the classification task.
            pretrained (bool): Whether to use pretrained weights for the transformer model.
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load the transformer model (e.g., BERT, RoBERTa) from Hugging Face
        self.encoder = AutoModel.from_pretrained(self.model_name)

        # If pretrained is False, initialize the weights randomly
        if not self.pretrained:
            self.encoder.init_weights()

        # Add a fully connected layer as the classification head
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, self.num_classes)

    def encode(self, input_ids, attention_mask):
        """
        Forward pass through the transformer encoder to extract embeddings.

        This method extracts the embeddings from the transformer encoder,
        specifically the hidden state corresponding to the first token for classification.

        Args:
            input_ids (Tensor): Input token IDs for the transformer model.
            attention_mask (Tensor): Attention mask to ignore padded tokens during model processing.

        Returns:
            Tensor: The embeddings corresponding to the first token in the sequence.
        """
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # [CLS] token representation

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the full model (transformer encoder + classification head).

        This method passes the input through the transformer encoder, 
        extracts the relevant embeddings, 
        and then applies the classification head to generate class logits.

        Args:
            input_ids (Tensor): Input token IDs for the transformer model.
            attention_mask (Tensor): Attention mask to handle padded tokens.

        Returns:
            Tensor: The logits for each class, corresponding to the classification task.
        """
        embeddings = self.encode(input_ids, attention_mask)  # Extract the embeddings
        logits = self.head(embeddings)  # Pass through the classification head
        return logits  # Output the logits for classification

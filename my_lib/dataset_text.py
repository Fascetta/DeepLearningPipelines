"""
Data processing module for creating a Hugging Face `Dataset` from a Pandas DataFrame.

This module defines a function `dataset_from_pandas` that converts a Pandas DataFrame into 
a Hugging Face `Dataset`. The DataFrame is preprocessed to combine object and description fields 
and then tokenized using a specified transformer model's tokenizer. The resulting dataset can 
be used for model training or evaluation in NLP tasks.

Functions:
    - dataset_from_pandas(df: pd.DataFrame, model_name: str, batched=True, test=False):
        Converts a Pandas DataFrame into a Hugging Face `Dataset` by preprocessing and tokenizing 
        the data using the specified transformer model.
    
    - preprocess(df: pd.DataFrame):
        Helper function to preprocess the DataFrame by combining object and description columns.
    
    - get_tokenizer(model_name: str):
        Helper function to load the tokenizer for the specified model.

    - tokenize_function(example, tokenizer, content_column="description", label_column="target"):
        Tokenizes the content and label columns in the DataFrame for model input.
"""
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd


def dataset_from_pandas(df: pd.DataFrame, model_name: str, batched=True, test=False):
    """
    Converts a Pandas DataFrame into a Hugging Face `Dataset` with tokenization.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        model_name (str): The name of the transformer model to use for tokenization.
        batched (bool, optional): Whether to batch the tokenization process. Defaults to True.
        test (bool, optional): If True, skips adding the 'label' column. Defaults to False.

    Returns:
        Dataset: A Hugging Face `Dataset` containing the tokenized data.

    The function preprocesses the DataFrame by combining the 'object' and 'description' columns 
    into a single column, and then tokenizes the resulting text using the specified tokenizer. 
    If not in test mode, it includes a 'label' column for model training or evaluation.
    """

    def preprocess(df: pd.DataFrame):
        """
        Preprocess the DataFrame by combining the 'object' and 'description' columns.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The processed DataFrame with a combined 'description' column.
        """
        df["description"] = df["object"] + ": " + df["description"]
        return df

    def get_tokenizer(model_name):
        """
        Load the tokenizer for the specified transformer model.

        Args:
            model_name (str): The name of the transformer model.

        Returns:
            AutoTokenizer: The tokenizer associated with the specified model.
        """
        return AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(
        example, tokenizer, content_column="description", label_column="target"
    ):
        """
        Tokenizes the input text from the DataFrame using the specified tokenizer.

        Args:
            example (dict): The data example to tokenize.
            tokenizer (AutoTokenizer): The tokenizer to use.
            content_column (str, optional): The column containing the text to tokenize.
            label_column (str, optional): The column containing the labels. Defaults to 'target'.

        Returns:
            dict: A dictionary containing the tokenized inputs and labels (if not in test mode).
        """
        out_dict = tokenizer(
            example[content_column], truncation=True, padding=True
        )  # truncation and padding ensure equal length sequences
        if not test:
            out_dict["label"] = example[label_column]
        return out_dict

    df = preprocess(df)
    tokenizer = get_tokenizer(model_name)

    return Dataset.from_pandas(df).map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=batched,
        batch_size=len(df),
    )

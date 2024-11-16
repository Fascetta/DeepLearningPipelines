# Project Documentation

## Overview

This project provides a complete pipeline for training deep learning models, particularly for visual and natural language processing (NLP) tasks. It includes utility functions, model architectures, configuration management, dataset handling, preprocessing routines, and reproducibility tools for research and experimentation.

## Features

- **Visual Model**: A flexible deep learning model using TIMM (pretrained models) for image classification.
- **NLP Model**: A transformer-based model for text classification, built on Hugging Face's transformers library.
- **Config Management**: Configuration classes for storing and loading hyperparameters and model settings.
- **Dataset Handling**: Utilities for processing datasets and loading them in batches, including image and text datasets.
- **Preprocessing**: Image preprocessing pipelines for training and testing.
- **Utility Functions**: Reproducibility, running mean calculation, and folder management.
  
## Requirements

This project requires Python 3.7+ and the following libraries:

```bash
pip install torch torchvision transformers timm datasets numpy pandas pillow
```

Additionally, depending on the environment, you might need to install CUDA for GPU support.

## Project Structure

``` graphql
project_root/
│
├── config_visual.py            # Configuration for visual model (hyperparameters, settings)
├── config_text.py              # Configuration for NLP model (hyperparameters, settings)
├── dataset_processing.py       # Dataset preparation, tokenization, transformations
├── models/                     # Folder containing model definitions (visual and NLP)
│   ├── visual_model.py         # Visual model using TIMM
│   ├── nlp_model.py           # NLP model using Hugging Face
├── utils/                      # Utility functions for reproducibility and folder management
│   ├── utils.py                # Helper functions (e.g., running mean, seed setting)
├── output/                     # Folder for saving model outputs and logs
└── README.md                   # Project documentation (this file)
```

## Configuration

The project uses configuration classes to manage hyperparameters and model settings, ensuring consistency and easy updates.

### Example VisualConfig (`config_visual.py`)

```python
@dataclass
class VisualConfig:
    # General parameters
    project_name: str = "project_visual"
    random_state: int = 42
    device: str = "cuda"
    seed: int = 42

    # Model configuration
    num_classes: int = 4
    model_name: str = "resnet18"
    pretrained: bool = True

    # Preprocessing parameters
    train_input_size: Tuple[int, int] = (224, 224)
    test_input_size: Tuple[int, int] = (224, 224)
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
```

### Example TextConfig (`config_text.py`)

```python
@dataclass
class TextConfig:
    project_name: str = "project_text"
    random_state: int = 42
    device: str = "cuda"
    seed: int = 42

    # Model parameters
    num_classes: int = 4
    model_name: str = "roberta-base"
    pretrained: bool = True
```

## Usage

### 1. **Model Training**

To train a model (visual or text), follow these steps:

1. **Define the configuration** for your model by creating an instance of the appropriate config class (`VisualConfig` or `TextConfig`).
2. **Prepare the dataset** using `dataset_from_pandas` for text data or `DFDataset` for image data.
3. **Define the model** (e.g., `VisualModelTimm` for visual tasks or `NLPClassificationModel` for NLP tasks).
4. **Train the model** by passing the data through the model and using your optimizer.

### 2. **Reproducibility**

To ensure reproducibility across runs, use the `seed_everything` function to set the random seed for PyTorch, numpy, and random libraries.

```python
from utils.utils import seed_everything

# Set a random seed for reproducibility
seed_everything(42)
```

### 3. **Folder Management**

Use `get_output_folder` to create unique folders based on project name and timestamp for storing logs, models, and results:

```python
from utils.utils import get_output_folder

output_folder = get_output_folder("project_visual")
print(output_folder)  # Example: 'project_visual_11-16-2024-10-23-45'
```

### 4. **Preprocessing**

For visual data, use the `get_preprocessing` function to apply a series of transformations like rotation, resizing, and normalization:

```python
from dataset_processing import get_preprocessing
from config_visual import VisualConfig

# Initialize configuration
config = VisualConfig()

# Get preprocessing function for training data
transform = get_preprocessing(config, is_training=True)
```

For text data, preprocessing is handled inside the `dataset_from_pandas` function which tokenizes and formats the input.

### Example Dataset Loading (Text)

```python
from dataset_processing import dataset_from_pandas
import pandas as pd

# Load your dataframe
df = pd.read_csv("your_dataset.csv")

# Preprocess and tokenize the dataset
dataset = dataset_from_pandas(df, model_name="roberta-base")
```

### Example Dataset Loading (Image)

```python
from dataset_processing import DFDataset
from config_visual import VisualConfig
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Define transformations
transform = get_preprocessing(VisualConfig())

# Create custom dataset
dataset = DFDataset(root="dataset/images", df=df, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Utility Functions

- **`RunningMean`**: Computes running means iteratively as values are received. Useful for tracking metrics like loss during training.
  
- **`seed_everything`**: Ensures deterministic results by setting random seeds across multiple libraries.

- **`get_output_folder`**: Creates a folder with a timestamp for storing experiment outputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Let me know if you want any adjustments or more details added to the README!

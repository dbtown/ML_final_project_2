"""
Utility Functions for EENG-645 Labs

This module provides helper functions for:
- Reproducibility (seeding)
- Weights & Biases integration
- Plotting learning curves and confusion matrices
- Model saving and loading with metadata
- NLP text processing (vocabulary, dataset, tokenization)

Students should NOT modify this file. It is imported by the main lab scripts.
"""

import itertools
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sklearn.metrics

from tqdm import tqdm

########################################################################################
# Reproducibility
########################################################################################


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python random.

    :param seed: The seed value to use (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


########################################################################################
# Weights & Biases Integration
########################################################################################

# Try to import wandb - will be None if not available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def init_wandb(
    project_name: str,
    run_name: str,
    config: dict = None,
    use_wandb: bool = True,
    group: str = None,
):
    """
    Initialize a Weights & Biases run for experiment tracking.

    :param project_name: Name of the wandb project
    :param run_name: Name for this specific run
    :param config: Dictionary of hyperparameters to log
    :param use_wandb: Whether to use wandb (set to False to disable)
    :param group: Group name for grouping related runs (e.g., trials in a search)
    :return: wandb run object or None if wandb is disabled/unavailable
    """
    if use_wandb and WANDB_AVAILABLE:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            group=group,
            reinit=True,
        )
        return run
    elif use_wandb and not WANDB_AVAILABLE:
        print("wandb not installed. Install with: pip install wandb")
    return None


def log_to_wandb(metrics: dict, step: int = None, use_wandb: bool = True):
    """
    Log metrics to Weights & Biases.

    :param metrics: Dictionary of metric names and values
    :param step: Optional step number (epoch)
    :param use_wandb: Whether to use wandb
    """
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(metrics, step=step)


def log_image_to_wandb(image_path: str, caption: str = None, use_wandb: bool = True):
    """
    Log an image file to Weights & Biases.

    :param image_path: Path to the image file
    :param caption: Optional caption for the image
    :param use_wandb: Whether to use wandb
    """
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({caption or image_path: wandb.Image(image_path, caption=caption)})


def finish_wandb(use_wandb: bool = True):
    """
    Finish the current wandb run.

    :param use_wandb: Whether wandb is being used
    """
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


########################################################################################
# Plotting Functions - Classification
########################################################################################


def plot_learning_curves_classification(
    results: dict,
    save_path: str = None,
):
    """
    Plot learning curves for multiple classification models.
    Shows both loss and accuracy curves.

    :param results: Dictionary with model names as keys, containing 'history' dict
                   with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists,
                   and optionally 'accuracy' or 'test_accuracy' for final metric.
    :param save_path: Path to save the figure (optional)
    :return: matplotlib figure
    """
    # Two rows: loss curves and accuracy curves
    fig, axes = plt.subplots(2, len(results), figsize=(5 * len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(2, 1)

    for idx, (name, data) in enumerate(results.items()):
        # Loss plot
        axes[0, idx].plot(data["history"]["train_loss"], label="Train Loss")
        axes[0, idx].plot(data["history"]["val_loss"], label="Val Loss")
        axes[0, idx].set_title(f"{name.capitalize()} Model - Loss")
        axes[0, idx].set_xlabel("Epoch")
        axes[0, idx].set_ylabel("Loss (BCE)")
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1, idx].plot(data["history"]["train_acc"], label="Train Acc")
        axes[1, idx].plot(data["history"]["val_acc"], label="Val Acc")
        metric_val = data.get(
            "accuracy",
            data.get(
                "test_accuracy",
                data.get("val_acc", data["history"].get("val_acc", [0])[-1]),
            ),
        )
        axes[1, idx].set_title(
            f"{name.capitalize()} Model - Accuracy\nVal/Test Acc: {metric_val:.4f}"
        )
        axes[1, idx].set_xlabel("Epoch")
        axes[1, idx].set_ylabel("Accuracy")
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nLearning curves saved to {save_path}")

    return fig


def plot_cm(
    cm: np.ndarray,
    classes: list,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap=None,
    save_path: str = None,
):
    """
    Plot a general confusion matrix for any number of classes.
    Adapted from sklearn examples.

    :param cm: Confusion matrix from sklearn.metrics.confusion_matrix
    :param classes: List of class names
    :param normalize: If True, normalize rows to sum to 1
    :param title: Title for the plot
    :param cmap: Colormap to use (default: plt.cm.Blues)
    :param save_path: Path to save the figure (optional)
    :return: matplotlib figure
    """
    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    if normalize:
        # Format the confusion matrix with 2 decimal places for better readability
        print(np.array2string(cm, formatter={"float_kind": lambda x: f"{x:.2f}"}))
    else:
        print(cm)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    return fig


########################################################################################
# Model Saving and Loading
# These are optional utilities to save models with metadata, and not required for lab 4
########################################################################################


def save_model_with_metadata(
    model: nn.Module,
    model_path: str,
    metadata: dict,
    save_full_model: bool = False,
):
    """
    Save a PyTorch model with metadata in multiple formats for autograding.

    This function saves THREE files:
    1. {model_path} - Checkpoint with state_dict + metadata (for reconstruction)
    2. {model_path}.full - Complete pickled model (for direct loading)
    3. {model_path}_metadata.json - JSON metadata (for easy inspection)

    :param model: The PyTorch model to save
    :param model_path: Path to save the model (should end in .pt)
    :param metadata: Dictionary containing model metadata
    :param save_full_model: Whether to save the complete pickled model
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 1. Save checkpoint with state dict + metadata (original format)
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        model_path,
    )

    # 2. Save complete pickled model (for direct loading in autograder)
    full_model_path = model_path + ".full"
    if save_full_model:
        torch.save(model, full_model_path)

    # 3. Save metadata as JSON for easy inspection
    metadata_path = model_path.replace(".pt", "_metadata.json")
    serializable_metadata = _make_json_serializable(metadata)
    with open(metadata_path, "w") as f:
        json.dump(serializable_metadata, f, indent=2)

    print(f"Model saved to {model_path}")
    if save_full_model:
        print(f"Full model saved to {full_model_path}")
    print(f"Metadata saved to {metadata_path}")


def load_model_with_metadata(model_class, model_path: str, input_size: int):
    """
    Load a PyTorch model along with its metadata.

    :param model_class: The class of the model to instantiate
    :param model_path: Path to the saved model checkpoint
    :param input_size: Input size for the model
    :return: Tuple of (model, metadata)
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = model_class(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = checkpoint.get("metadata", {})
    return model, metadata


def load_model_full(model_path: str) -> nn.Module:
    """
    Load a complete pickled model directly.

    This is useful for loading models when you don't have the model class available.

    :param model_path: Path to the .pt file (will automatically try .full version)
    :return: The complete PyTorch model
    """
    full_path = model_path + ".full" if not model_path.endswith(".full") else model_path
    if os.path.exists(full_path):
        model = torch.load(full_path, map_location="cpu", weights_only=False)
        return model

    raise FileNotFoundError(
        f"Full model not found at {full_path}. "
        "Ensure save_model_with_metadata() was used to save the model."
    )


def load_metadata_json(model_path: str) -> dict:
    """
    Load metadata from the JSON sidecar file.

    :param model_path: Path to the .pt model file
    :return: Metadata dictionary, or None if not found
    """
    json_path = model_path.replace(".pt", "_metadata.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def _make_json_serializable(obj):
    """Convert numpy types and other non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


########################################################################################
# Lab 6: NLP / Text Classification Utilities
########################################################################################


class FanFicDataset(Dataset):
    """
    PyTorch Dataset for fan fiction text files organised in class subdirectories.

    Expected layout::

        data_dir/
            class_a/
                file1.txt
                file2.txt
            class_b/
                file3.txt
                ...

    All files are read into memory at init time (fine for the small dataset).

    :param data_dir: Root directory containing one subfolder per class
    :param tokenizer: A trained ``tokenizers.Tokenizer`` with padding and
                      truncation already enabled
    """

    def __init__(self, data_dir: str, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []  # list of (token_ids, label)
        self.class_names = sorted(  # sort for consistency
            [  # list comprehension to find subdirectories as class names
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )
        # dictionary comprehension to map class names to integer labels (0, 1, 2, ...)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        print(f"Loading dataset from {data_dir} with classes: {self.class_names}")
        for class_name in tqdm(self.class_names, desc="Classes"):
            # construct path to class subdirectory and get label index
            class_dir = os.path.join(data_dir, class_name)
            # set label/target to index of class name in sorted list of class names
            label = self.class_to_idx[class_name]
            for fname in os.listdir(class_dir):
                # only read .txt files, skip others (e.g. .DS_Store)
                if fname.endswith(".txt"):
                    fpath = os.path.join(class_dir, fname)
                    try:
                        # open file with utf-8 encoding and ignore errors, read text
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        # use provided tokenizer to encode text to token ids
                        encoded = tokenizer.encode(text)
                        # append (token_ids, label) tuple to samples list for
                        # later __getitem__
                        self.samples.append((encoded.ids, label))
                    except Exception as e:
                        print(f"Warning: Could not read {fpath}: {e}")

        print(
            f"Loaded {len(self)} samples from {data_dir} "
            f"({len(self.class_names)} classes: {self.class_names})"
        )

    def __len__(self):
        """Parent class suggests implementing this."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Parent class requires implementing this to return
        (input, label) for a given index.
        Here we return (token_ids, label) as tensors."""
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


class BaselineModel(nn.Module):
    """
    Provided dense baseline model that operates on word-count vectors.
    Achieves ~85 % accuracy.  Students should implement ``RNNModel`` to beat it.

    :param vocab_size: Size of the vocabulary (input dimension)
    :param num_classes: Number of output classes
    :param hidden_size: Hidden layer width
    :param dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 50,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Do NOT look at this code for reference when implementing RNNModel!
        This does some hacky stuff to use only Linear layers and is different for RNNs.
        """
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        # AveragePool1d basically
        pooled = embedded.mean(dim=1)  # (batch, emb_dim)
        return self.classifier(pooled)  # (batch, num_classes)


def get_predictions(model: nn.Module, dataloader, device) -> tuple:
    """
    Run *model* over every batch in *dataloader* and collect predictions.

    :param model: Trained PyTorch model
    :param dataloader: DataLoader to iterate over
    :param device: Device the model lives on
    :return: ``(y_true, y_pred)`` as numpy arrays
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def visualize_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    model_name: str = "Model",
    fig_folder: str = "./figures",
):
    """
    Visualize predictions using classification report and confusion matrix.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param class_names: List of class names
    :param model_name: Name for the model (used in titles)
    :param fig_folder: Folder to save figures
    """
    os.makedirs(fig_folder, exist_ok=True)

    # Print classification report
    print(f"\n{'='*60}")
    print(f"Classification Report for {model_name}")
    print("=" * 60)
    print(
        sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)
    )

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Plot confusion matrices
    plot_cm(
        cm,
        classes=class_names,
        normalize=False,
        title=f"{model_name} - Confusion Matrix",
        save_path=os.path.join(fig_folder, f"{model_name}_confusion_matrix.png"),
    )

    plot_cm(
        cm,
        classes=class_names,
        normalize=True,
        title=f"{model_name} - Normalized Confusion Matrix",
        save_path=os.path.join(
            fig_folder, f"{model_name}_confusion_matrix_normalized.png"
        ),
    )

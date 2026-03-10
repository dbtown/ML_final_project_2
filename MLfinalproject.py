"""
ML Final Project: Orbit Prediction with Machine Learning (GRU Cells)
Author: David Brown

Description:
This project focuses on predicting satellite orbits using machine learning techniques. 
We will utilize a dataset containingtime series of orbital parameters and radial velocity 
measurements for a set of satellites. The goal is to train a model that can accurately 
predict future orbital states based on historical data.

This project follows Chollet's 7-step machine learning workflow, which includes:
1. Defining the problem and assembling the data
2. Preparing the data
3. Developing a model
4. Training the model
5. Evaluating the model
6. Tuning the model
7. Presenting the results


"""

#General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

#Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

#Hyperparameter tuning
import optuna
from optuna.integration import WeightsAndBiasesCallback

#Weights & Biases
import wandb

# Flags and constants
USE_WANDB = False  # Set to True to use Weights & Biases for experiment tracking
WANDB_PROJECT_NAME = "ML Final Project - Orbit Prediction Using GRU Cells"
OUTPUT_TYPE = "rv"  # "coe" for classical orbital elements, "rv" for radial velocity data
RANDOM_SEED = 42
USE_TEST_SET = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Search space for hyperparameter tuning
NUM_SAMPLES = 10
MAX_EPOCHS = 15



# ============================================================================
# Step 1: Assemble the Data
# ============================================================================
# Done.


# ============================================================================
# Step 2: Data Wrangling
# ============================================================================
# Load the dataset
data_path = Path(f"./data new/{OUTPUT_TYPE}_orbit_300164_timeseries.csv")

def load_and_prepare_orbit_data(data_path):
    """
    Load and prepare the orbit dataset for training. The labels will be the future states, while the 
    features will be the current time. This function also normalizes the features that need it and 
    splits the data into training, validation, and test splits as well as constructs the datasets 
    and loaders. Since this is a time series dataset, the split will be done sequentially, not randomly. 
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # No cleaning needed except normalization, as well as no feature engineering needed. However, 
    # the time feature needs to be converted from datetime to a numeric value.
    df["time"] = pd.to_datetime(df["time"])
    t0 = df["time"].iloc[0]
    df["t_seconds"] = (df["time"] - t0).dt.total_seconds()


    # Normalize
    X = df[["t_seconds"]].values # Time is the feature
    y = df.drop("time", axis=1).values # The coes or rv values are the labels
    feature_names = df.drop("time", axis=1).columns.tolist()
    X_scaler = MinMaxScaler() # Time is a large value since its 6 years of data and the time step should be linear, plus normalizing it between 0 and 1 is better for vanishing gradient issues.
    X_scaled = X_scaler.fit_transform(X)
    y_scaler = StandardScaler() # StandarScaler should work for coes and rv values.
    y_scaled = y_scaler.fit_transform(y)

    # Time to split the data into train, valid, and test sets. I will use a 75-15-15 split.
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.15, shuffle = False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.1765, shuffle = False)

    # Construct datasets
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype = torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype = torch.float32))

    return train_dataset, val_dataset, test_dataset, feature_names

def construct_dataloaders(train_dataset, val_dataset, test_dataset, batch_size: int = 32):
    """
    Constructs dataloaders in the proper formatting.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================================
# Step 3: Develop Model
# ============================================================================
class GRUPredictor(nn.Module):
    """
    RNN-based predictor using GRU layers.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float = 0.0
    ):
        super(GRUPredictor, self).__init__()
        self.config = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout
        }
        self.gru = nn.GRU(input_size, self.config["hidden_size"], self.config["num_layers"], batch_first=True, dropout=self.config["dropout"])
        self.fc = nn.Linear(self.config["hidden_size"], 6)

    def forward(self, x):
        output, hidden = self.gru(x)
        output = self.fc(output[:,-1,:])

        return output
    

# ============================================================================
# Step 4: Train the Model
# ============================================================================
def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device
):
    """
    Trains model for one epoch
    """
    model.train()
    running_total_loss = 0.0
    correct = 0.0
    total_predictions = 0.0

    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_batch).sum().item()


# ============================================================================
# Step 5: Evaluate the Model
# ============================================================================

# ============================================================================
# Step 6: Tune the Model
# ============================================================================

# ============================================================================
# Step 7: Present Results
# ============================================================================





    






"""
ML Final Project: Orbit Prediction with Machine Learning (GRU Cells)
Author: David Brown

Description:
This project focuses on predicting satellite orbits using machine learning techniques. 
We will utilize a dataset containingtime series of orbital parameters and radial velocity 
measurements for a set of satellites. The goal is to train a model that can accurately 
predict future orbital states based on historical data.

This project follows Chollet's 7-step machine learning workflow, which includes:
1. Data Collection
2. Preparation
3. Choosing a Model
4. Training
5. Evaluation
6. Tuning the model
7. Deployment/Results

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
NUM_SEQ = 20

# Search space for hyperparameter tuning
NUM_SAMPLES = 10
MAX_EPOCHS = 15



# ============================================================================
# Step 1: Data Collection
# ============================================================================
# Done.


# ============================================================================
# Step 2: Data Preparation
# ============================================================================
# Load the dataset
data_path = Path(f"./data new/{OUTPUT_TYPE}_orbit_300164_timeseries.csv")

def load_and_prepare_orbit_data(data_path, NUM_SEQ):
    """
    Load and prepare the orbit dataset for training. The labels will be the future states, while the 
    features will be the current time. This function also normalizes the features that need it and 
    splits the data into training, validation, and test splits as well as constructs the datasets 
    and loaders. Since this is a time series dataset, the split will be done sequentially, not randomly. 
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    if OUTPUT_TYPE == "coe":
        feature_names = ["Semimajor Axis", "Eccentricity", "Inclination", "RAAN", "Argument of Perigee", "True Anomaly"]
    if OUTPUT_TYPE == "rv":
        feature_names = ["Rx", "Ry", "Rz", "Vx", "Vy", "Vz"]
    else:
        raise ValueError("Invalid OUTPUT_TYPE. Must be 'coe' or 'rv'.")
    states = df[feature_names].values

    # No cleaning needed except normalization, as well as no feature engineering needed.

    # Normalize the features and labels
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)


    # Create time series sequences
    def create_sequences(states, num_seq):
        X = []
        y = []
        for i in range(len(states) - num_seq):
            X.append(states[i:i+num_seq])
            y.append(states[i+num_seq])
        return np.array(X), np.array(y)
    
    X_seq, y_seq = create_sequences(states_scaled, NUM_SEQ)

    # Time to split the data into train, valid, and test sets. I will use a 75-15-15 split.
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_seq, y_seq, test_size = 0.15, shuffle = False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.1765, shuffle = False)

    # Construct datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype = torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype = torch.float32))

    return train_dataset, val_dataset, test_dataset, feature_names

def construct_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Constructs dataloaders in the proper formatting.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================================
# Step 3: Choose Model
# ============================================================================
class GRUPredictor(nn.Module):
    """
    RNN-based predictor using GRU layers.
    """

    def __init__(
            self,
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

        self.gru = nn.GRU(6, self.config["hidden_size"], self.config["num_layers"], batch_first=True, dropout=self.config["dropout"])
        self.fc = nn.Linear(self.config["hidden_size"], 6)

    def forward(self, x):
        output, hidden = self.gru(x)
        output = self.fc(output[:,-1,:])

        return output
    
def create_model(config: Dict[str, Any]):
    """
    Create model from a dictionary
    """
    model = GRUPredictor(
        hidden_size = config["hidden_size"],
        num_layers = config["num_layers"],
        dropout = config["dropout"]
    )
    return model


# ============================================================================
# Step 4: Training
# ============================================================================
def train_epoch(
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
    train_loss = 0.0

    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(train_loader)

# ============================================================================
# Step 5: Evaluation
# ============================================================================
def evaluate(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
):
    """
    Evaluation loop
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# ============================================================================
# Step 6: Tune the Model
# ============================================================================
def get_search_space_optuna(trial: optuna.Trial):
    """
    Define search space for optuna search
    """
    config = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])
    }
    return config

def optuna_objective(trial: optuna.Trial):
    """
    Optuna obj func
    """
    config = get_search_space_optuna(trial)
    train_dataset, val_dataset = load_and_prepare_orbit_data(data_path, NUM_SEQ)
    train_loader, val_loader = construct_dataloaders(train_dataset, val_dataset, batch_size = config["batch_size"])
    model = create_model(config)
    model = model.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    best_val_acc = 0.0
    progress_bar = tqdm(range(MAX_EPOCHS), desc = f"Trial {trial.number}", leave = False)
        
    for step in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)

        best_val_acc = min(val_loss)
        progress_bar.set_postfix(


        
    trial.report(
    if trial.should_prune():
        raise optuna.TrialPruned()

return best_val_acc

                           
    



# ============================================================================
# Step 7: Present Results
# ============================================================================





    






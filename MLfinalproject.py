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
from scipy.integrate import solve_ivp
from tqdm import tqdm

#Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

#Hyperparameter tuning
import optuna
from optuna.integration import WeightsAndBiasesCallback

#Weights & Biases
import wandb

# Flags
USE_WANDB = True  # Set to True to use Weights & Biases for experiment tracking
USE_TEST_SET = False
OUTPUT_TYPE = "rv"  # "coe" for classical orbital elements, "rv" for radial velocity data
RUN_BASELINE = True
RUN_MAIN = False

# Constants
WANDB_PROJECT_NAME = "ML Final Project - Orbit Prediction Using GRU Cells"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Configuration
NUM_SEQ = 20 #How many steps is the training window
PRED_STEPS = 20 ######How many steps to predict, if you change this you have to change GRU output size as well!!!!


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

def load_and_prepare_orbit_data(data_path, NUM_SEQ, PRED_STEPS):
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


    # Create time series sequences (num seq is how many in the past, pred steps is how many in the future)
    def create_sequences(states, num_seq, pred_steps):
        X = []
        y = []
        for i in range(len(states) - num_seq - pred_steps):
            X.append(states[i:i+num_seq])
            y.append(states[i+num_seq:i+num_seq+pred_steps])
        return np.array(X), np.array(y)
    
    X_seq, y_seq = create_sequences(states_scaled, NUM_SEQ, PRED_STEPS)

    # Time to split the data into train, valid, and test sets. I will use a 75-15-15 split.
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_seq, y_seq, test_size = 0.15, shuffle = False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.1765, shuffle = False)

    # Construct datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype = torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype = torch.float32))

    return train_dataset, val_dataset, test_dataset, feature_names, scaler

def construct_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Constructs dataloaders in the proper formatting.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================================
# BASELINE: CR3BP Numerical Integration
# ============================================================================
# pbar = tqdm(total=100, desc="Integrating Orbit", unit="%")
# last_pct = 0

# def cr3bp_eom(t, state, t_final):
#     #progress bar
#     global last_pct
#     current_pct = int((t/t_final)*100)
#     if current_pct > last_pct:
#         pbar.update(current_pct-last_pct)
#         last_pct = current_pct

#     #Constants and EOMs
#     mu = 0.0121505856
#     x,y,z,vx,vy,vz = state
#     r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
#     r2 = np.sqrt((x - (1-mu))**2 + y**2+z**2)

#     ax = 2*vy+x- (1-mu)*(x+mu)/r1**3 - mu*(x-(1-mu))/r2**3

#     ay = -2*vx+y - (1-mu)*y/r1**3 - mu*y/r2**3

#     az = -(1-mu)*z/r1**3 - mu*z/r2**3

#     return np.array([vx,vy,vz,ax,ay,az])
    
        
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
        self.fc = nn.Linear(self.config["hidden_size"], 6*20)

    def forward(self, x):
        output, hidden = self.gru(x)
        output = self.fc(output[:,-1,:])
        output = output.view(-1, 20, 6)

        return output
    
def create_model(config):
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
    train_loss /= len(train_loader)
    train_rmse = np.sqrt(train_loss)
    return train_loss, train_rmse

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
    val_loss /= len(val_loader)
    val_rmse = np.sqrt(val_loss)
    return val_loss, val_rmse

# ============================================================================
# Step 6: Tune the Model
# ============================================================================
def get_search_space(trial: optuna.Trial):
    """
    Define search space for optuna search
    """
    config = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])
    }
    return config

def search_objective(trial: optuna.Trial):
    """
    Optuna obj func
    """
    config = get_search_space(trial)
    train_dataset, val_dataset, _, _, _ = load_and_prepare_orbit_data(data_path, NUM_SEQ)
    train_loader, val_loader, _ = construct_dataloaders(train_dataset, val_dataset, _, batch_size = config["batch_size"])
    model = create_model(config)
    model = model.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    best_val_rmse = 10000.0
    progress_bar = tqdm(range(MAX_EPOCHS), desc = f"Trial {trial.number}", leave = False)
        
    for step in progress_bar:
        train_loss, train_rmse = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, DEVICE)

        best_val_rmse = min(best_val_rmse, val_rmse)
        progress_bar.set_postfix({"val_rmse": f"{val_rmse:.4f}", "new best": f"{best_val_rmse:.4f}"})

    trial.report(val_rmse, step)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return best_val_rmse

def run_search(n_trials: int = NUM_SAMPLES):
    """
    Run the search! Direction is minimize because we are working with RMSE.
    """
    # Create the study
    study = optuna.create_study(direction="minimize", pruner = optuna.pruners.MedianPruner(n_startup_trials = 2, n_warmup_steps=3))

    # WandB
    callbacks =[]
    if USE_WANDB:
        wandb_callback = WeightsAndBiasesCallback(
            metric_name = "best_val_RMSE",
            wandb_kwargs = {"project": WANDB_PROJECT_NAME},
            as_multirun = True
        )
        callbacks.append(wandb_callback)
    # Optimization
    study.optimize(search_objective, n_trials = n_trials, callbacks=callbacks, show_progress_bar=True)

    print(f"\nBest RMSE: {study.best_trial.value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study

# ============================================================================
# Baseline Stuff
# ============================================================================
def two_body_j2(t, state):
    MU = 398600.4418e9
    RE = 6378137.0
    J2 = 1.08262668e-3
    r = state[0:3]
    v = state[3:6]

    x,y,z= r
    r_norm = np.linalg.norm(r)

    a_2b = -MU * r/r_norm**3

    zx = z/r_norm
    factor = (3/2)*J2*MU*RE**2/r_norm**5

    ax = factor*x*(5*zx**2-1)
    ay = factor*y*(5*zx**2-1)
    az = factor*z*(5*zx**2-3)

    a_j2 = np.array([ax,ay,az])

    a_total = a_2b+a_j2

    return np.hstack((v, a_total))
def prop_20_steps(initial_state, dt, steps=20):
    t_span = (0, steps*dt)
    t_eval = np.linspace(0, steps*dt, steps+1)

    sol = solve_ivp(
        two_body_j2,
        t_span,
        initial_state,
        t_eval,
        rtol = 1e-9,
        atol = 1e-9
    )
    future_states = sol.y.T[1:]

def visualize_predictions(model, val_loader, scaler, dt):
    model.eval()

    X_batch, y_batch = next(iter(val_loader))
    X_example = X_batch[0:1].to(DEVICE)

    truth_future = y_batch[0].cpu().numpy()

    with torch.no_grad():
        pred_future = model(X_example)
    pred_future.cpu().numpy()[0]

    pred_future = scaler.inverse_transform(pred_future)
    truth_future = scaler.inverse_transform(truth_future)

    initial_states_scaled = X_example[0,-1,:].cpu().numpy()
    initial_state = scaler.inverse_transform(initial_states_scaled.reshape(1,-1))[0]

    baseline_future = prop_20_steps(initial_state,dt)

    truth_r = truth_future[:,:3]
    gru_r = pred_future[:,:3]
    baseline_r = baseline_future[:,:3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth_r[:,0], truth_r[:,1], truth_r[:,2], label="Truth", linewidth=3)
    ax.plot(gru_r[:,0], gru_r[:,1], gru_r[:,2], label="GRU Prediction")
    ax.plot(baseline_r[:,0], baseline_r[:,1], baseline_r[:,2], label="Two-body +J2")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    plt.show()

    gru_error = np.linalg.norm(gru_r-truth_r, axis=1)
    baseline_error = np.linalg.norm(baseline_r-truth_r, axis=1)

    t = np.arange(len(gru_error))

    plt.figure()
    plt.plot(t, gru_error, label = "GRU Error")
    plt.plot(t, baseline_error, label = "Two Body + J2 Error")
    plt.xlabel("Future Timestep")
    plt.ylabel("Position error (m)")

    plt.legend()
    plt.show()

# ============================================================================
# Step 7: Present Results
# ============================================================================

def main():
    """
    Run the whole thing wooo
    """
    optuna_study = run_search(n_trials=NUM_SAMPLES)
    best_params = optuna_study.best_trial.params

    if RUN_BASELINE:
        train_ds, val_ds, test_ds, _, scaler = load_and_prepare_orbit_data(data_path, NUM_SEQ)
        _, val_loader, _ = construct_dataloaders(train_ds, val_ds, test_ds, best_params["batch_size"])
        best_config = {
            "hidden_size": best_params["hidden_size"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"]
        }
        
        model = create_model(best_config).to(DEVICE)

        dt = 3599.178006
        visualize_predictions(model, val_loader, scaler, dt)

    if USE_TEST_SET:
        train_ds, val_ds, test_ds, _, _= load_and_prepare_orbit_data(data_path, NUM_SEQ)
        _, _, test_loader = construct_dataloaders(train_ds, val_ds, test_ds, best_params["batch_size"])
        best_config = {
            "hidden_size": best_params["hidden_size"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"]
        }
        
        best_model = create_model(best_config).to(DEVICE)

        # Testing eval
        criterion = nn.MSELoss()
        test_loss, test_rmse = evaluate(best_model, test_loader, criterion, DEVICE)
        print(f"Test RMSE: {test_rmse:.4f}")

if RUN_MAIN:
    main()

# ============================================================================
# Step 8: Baseline Physics Model
# ============================================================================
















# if RUN_BASELINE:
#     df_all = pd.read_csv(data_path)
#     test_start_idx = 44709 #either 44709 or 44710, 44709.15
#     df = df_all.iloc[test_start_idx:]
#     df["Time"] = pd.to_datetime(df["Time"])

#     actual_states = df.iloc[:,1:7]

#     #Time steps
#     t_start = df["Time"].iloc[0]
#     t_eval = (df['Time']-t_start).dt.total_seconds().values
#     t_span = (t_eval[0], t_eval[-1])
#     t_final = t_eval[-1]
#     t_eval = np.linspace(0,t_final, 5)
#     # time_seconds = np.linspace(0, total_seconds, N)


#     state0 = actual_states.iloc[0,0:6]

#     print(f"Integrating trajectory for {len(t_eval)} time steps")
#     sol = solve_ivp(
#         cr3bp_eom,
#         [0, t_final],
#         state0,
#         args=(t_final,),
#         rtol = 1e-8,
#         atol = 1e-8
#     )
#     pbar.close()

#     predicts = sol.y.T
#     mse = np.mean((actual_states-predicts)**2)
#     rmse = np.sqrt(mse)

#     print(f"Baseline RAN!!")
#     print(f"Final RMSE: {rmse:.4f}")




    






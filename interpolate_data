import pandas as pd
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from datetime import timedelta
from pathlib import Path


def expand_orbital_data(input_path, output_path, step_seconds=60):
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Ensure Time is in datetime format
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 1. Convert Time to numeric (seconds from the very first data point)
    start_time = df['Time'].iloc[0]
    t_numeric = (df['Time'] - start_time).dt.total_seconds().values
    
    # 2. Extract Position (R) and Velocity (V) vectors
    # CubicHermiteSpline uses 'y' for position and 'dydx' for velocity
    R_vectors = df[['Rx', 'Ry', 'Rz']].values
    V_vectors = df[['Vx', 'Vy', 'Vz']].values
    
    print("Generating Hermite Spline (enforcing velocity as the derivative)...")
    # This creates a mathematical function that "knows" the physics of the trajectory
    interp_func = CubicHermiteSpline(x=t_numeric, y=R_vectors, dydx=V_vectors)
    
    # 3. Create the new high-resolution time grid
    # np.arange(start, stop, step)
    t_new = np.arange(t_numeric[0], t_numeric[-1], step_seconds)
    
    print(f"Interpolating {len(t_new)} points...")
    R_new = interp_func(t_new)                # New Positions
    V_new = interp_func.derivative()(t_new)   # New Velocities
    
    # 4. Reconstruct the Timestamps
    # Convert numeric seconds back into actual DateTimes
    times_new = [start_time + timedelta(seconds=s) for s in t_new]
    
    # 5. Build the expanded DataFrame
    df_expanded = pd.DataFrame({
        'Time': times_new,
        'Rx': R_new[:, 0],
        'Ry': R_new[:, 1],
        'Rz': R_new[:, 2],
        'Vx': V_new[:, 0],
        'Vy': V_new[:, 1],
        'Vz': V_new[:, 2]
    })
    
    print(f"Saving to {output_path}...")
    df_expanded.to_csv(output_path, index=False)
    print("Expansion complete!")

# Run the process
expand_orbital_data(
    input_path= Path(f"./data new/rv_orbit_300164_timeseries.csv"), 
    output_path= Path(f"./data new/rv_orbit_300164_interpolated.csv"), 
    step_seconds=600  # This creates 10 minute steps
)
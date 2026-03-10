import h5py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =====================================================
# USER SETTINGS
# =====================================================

hdf5_file = "data new\orb_id_300150_to_300199 (1).h5"
group_name = "300164"


# Choose output mode
# Options: "coe" or "rv"
OUTPUT_MODE = "coe"

output_csv = f"data new\{OUTPUT_MODE}_orbit_{group_name}_timeseries.csv"
start_date = datetime(1980, 1, 1, 0, 0, 0)

# =====================================================
# READ DATA
# =====================================================

with h5py.File(hdf5_file, "r") as f:

    group = f[group_name]

    # lifetime stored in years
    lifetime_years = group["lifetime"][()]

    if OUTPUT_MODE == "coe":

        sma  = group["semi_major_axis"][:].flatten()
        ecc  = group["eccentricity"][:].flatten()
        inc  = group["inclination"][:].flatten()
        raan = group["longitude_of_ascending_node"][:].flatten()
        argp = group["argument_of_periapsis"][:].flatten()
        ta   = group["true_anomaly"][:].flatten()

        N = len(sma)

    elif OUTPUT_MODE == "rv":

        r = group["r"][:]   # Nx3
        v = group["v"][:]   # Nx3

        rx = r[:,0]
        ry = r[:,1]
        rz = r[:,2]

        vx = v[:,0]
        vy = v[:,1]
        vz = v[:,2]

        N = len(rx)

    else:
        raise ValueError("OUTPUT_MODE must be 'coe' or 'rv'")

# =====================================================
# CREATE TIME VECTOR
# =====================================================

seconds_per_year = 365.25 * 24 * 3600
total_seconds = lifetime_years * seconds_per_year

time_seconds = np.linspace(0, total_seconds, N)

time_vector = [start_date + timedelta(seconds=float(t)) for t in time_seconds]

# =====================================================
# BUILD DATAFRAME
# =====================================================

if OUTPUT_MODE == "coe":

    df = pd.DataFrame({
        "Time": time_vector,
        "Semimajor Axis": sma,
        "Eccentricity": ecc,
        "Inclination": inc,
        "RAAN": raan,
        "Argument of Perigee": argp,
        "True Anomaly": ta
    })

elif OUTPUT_MODE == "rv":

    df = pd.DataFrame({
        "Time": time_vector,
        "Rx": rx,
        "Ry": ry,
        "Rz": rz,
        "Vx": vx,
        "Vy": vy,
        "Vz": vz
    })

# =====================================================
# EXPORT CSV
# =====================================================

df.to_csv(output_csv, index=False)

print("CSV written:", output_csv)
print("Mode:", OUTPUT_MODE)
print("Samples:", N)
print("Lifetime (years):", lifetime_years)
print("Start:", time_vector[0])
print("End:", time_vector[-1])
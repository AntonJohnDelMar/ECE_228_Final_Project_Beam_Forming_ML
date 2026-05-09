import numpy as np
import pandas as pd


data = np.load("outputs/rx_sweep_result.npz", allow_pickle=True)

rx_positions = data["rx_positions"]
all_tau = data["all_tau"]
all_mag = data["all_mag"]
frequency = np.array(data["frequency"]).flatten()[0]

rows = []

for i, rx_pos in enumerate(rx_positions):
    tau = np.array(all_tau[i]).flatten()
    mag = np.array(all_mag[i]).flatten()

    if len(mag) > 0:
        strongest_idx = np.argmax(mag)
        strongest_delay_ns = tau[strongest_idx] / 1e-9
        strongest_mag = mag[strongest_idx]
        total_power = np.sum(np.abs(mag) ** 2)
        total_power_db = 10 * np.log10(total_power + 1e-30)
    else:
        strongest_delay_ns = np.nan
        strongest_mag = 0.0
        total_power = 0.0
        total_power_db = -300.0

    row = {
        "sample_id": i,
        "rx_x": rx_pos[0],
        "rx_y": rx_pos[1],
        "rx_z": rx_pos[2],
        "num_paths": len(mag),
        "strongest_delay_ns": strongest_delay_ns,
        "strongest_mag": strongest_mag,
        "total_power": total_power,
        "total_power_db": total_power_db,
        "frequency_hz": frequency,
    }

    rows.append(row)

df = pd.DataFrame(rows)

df.to_csv("outputs/rx_sweep_features.csv", index=False)

print(df)
print("\nSaved outputs/rx_sweep_features.csv")
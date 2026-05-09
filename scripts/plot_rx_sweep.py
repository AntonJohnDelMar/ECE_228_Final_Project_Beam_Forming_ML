import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("outputs/rx_sweep_features.csv")

plt.figure()
plt.plot(df["rx_x"], df["total_power_db"], marker="o")
plt.xlabel("RX x-position [m]")
plt.ylabel("Total received power [dB]")
plt.title("Received Power vs RX Position")
plt.grid(True)
plt.savefig("outputs/rx_sweep_power.png", dpi=200)
plt.show()

print("Saved outputs/rx_sweep_power.png")
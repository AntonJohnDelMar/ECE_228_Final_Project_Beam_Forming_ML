import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("outputs/dataset_v3.csv")

plt.figure()
plt.scatter(df["u1_x"], df["u1_y"], c=df["tx0_sector_label"])
plt.xlabel("User 1 x-position [m]")
plt.ylabel("User 1 y-position [m]")
plt.title("User 1 Positions Colored by TX0 Sector Label")
plt.colorbar(label="TX0 sector")
plt.axis("equal")
plt.grid(True)
plt.savefig("outputs/user1_sector_labels.png", dpi=200)
plt.show()

plt.figure()
plt.scatter(df["u2_x"], df["u2_y"], c=df["tx1_sector_label"])
plt.xlabel("User 2 x-position [m]")
plt.ylabel("User 2 y-position [m]")
plt.title("User 2 Positions Colored by TX1 Sector Label")
plt.colorbar(label="TX1 sector")
plt.axis("equal")
plt.grid(True)
plt.savefig("outputs/user2_sector_labels.png", dpi=200)
plt.show()

angle_min = -65
angle_max = 65

plt.figure()
plt.scatter(df["u1_angle_deg"], df["tx0_target_angle_deg"], s=12)
plt.plot([angle_min, angle_max], [angle_min, angle_max], "k--", linewidth=1)
plt.xlabel("User 1 angle [deg]")
plt.ylabel("TX0 target angle [deg]")
plt.title("User 1 Angle vs Selected TX0 Target Angle")
plt.xlim(angle_min, angle_max)
plt.ylim(angle_min, angle_max)
plt.grid(True)
plt.savefig("outputs/user1_angle_vs_tx0_target_diag.png", dpi=200)
plt.show()

plt.figure()
plt.scatter(df["u2_angle_deg"], df["tx1_target_angle_deg"], s=12)
plt.plot([angle_min, angle_max], [angle_min, angle_max], "k--", linewidth=1)
plt.xlabel("User 2 angle [deg]")
plt.ylabel("TX1 target angle [deg]")
plt.title("User 2 Angle vs Selected TX1 Target Angle")
plt.xlim(angle_min, angle_max)
plt.ylim(angle_min, angle_max)
plt.grid(True)
plt.savefig("outputs/user2_angle_vs_tx1_target_diag.png", dpi=200)
plt.show()

df["u1_angle_error_deg"] = df["tx0_target_angle_deg"] - df["u1_angle_deg"]

plt.figure()
plt.hist(df["u1_angle_error_deg"], bins=40)
plt.xlabel("TX0 target angle - User 1 angle [deg]")
plt.ylabel("Count")
plt.title("User 1 Target Angle Error")
plt.grid(True)
plt.savefig("outputs/user1_angle_error_hist.png", dpi=200)
plt.show()

plt.hist(df["target_sinr_u1_db"], bins=40)
plt.hist(df["target_sinr_u2_db"], bins=40)
plt.hist(df["target_sum_rate_bpshz"], bins=40)

print("TX0 sector counts")
print(df["tx0_sector_label"].value_counts().sort_index())

print("TX0 codebook counts")
print(df["tx0_codebook_label"].value_counts().sort_index())

print("TX1 sector counts")
print(df["tx1_sector_label"].value_counts().sort_index())

print("TX1 codebook counts")
print(df["tx1_codebook_label"].value_counts().sort_index())
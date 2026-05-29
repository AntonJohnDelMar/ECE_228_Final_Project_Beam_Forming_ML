#dataset verification
import pandas as pd

df = pd.read_csv("outputs/dataset_8sector_3book_full_100.csv")

print(df.shape)
df.head()

print(df["target_is_best"].mean())

df["score_gap"] = df["best_score"] - df["target_score"]

print(df["score_gap"].describe())

print("\nOnly non-best samples:")
print(df[df["target_is_best"] == 0]["score_gap"].describe())
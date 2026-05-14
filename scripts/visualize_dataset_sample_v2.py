import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "outputs/raynet_mumimo_dataset_v2.csv"
METADATA_PATH = "outputs/raynet_mumimo_dataset_v2_metadata.json"

SAMPLE_ID = 0
USE_RANDOM_SAMPLE = False
RANDOM_SEED = 7

BEAM_SCALE_MARGIN = 1.15


def load_tx_position():
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        tx_position = metadata["config"]["tx_position"]
        return float(tx_position[0]), float(tx_position[1])
    except Exception:
        return 0.0, 0.0


def endpoint_from_angle(tx_x, tx_y, angle_deg, length):
    angle_rad = np.deg2rad(angle_deg)
    x_end = tx_x + length * np.cos(angle_rad)
    y_end = tx_y + length * np.sin(angle_rad)
    return x_end, y_end


def draw_beam(ax, tx_x, tx_y, angle_deg, length, label, color):
    x_end, y_end = endpoint_from_angle(tx_x, tx_y, angle_deg, length)

    ax.plot(
        [tx_x, x_end],
        [tx_y, y_end],
        linewidth=2.5,
        color=color,
        label=label,
    )

    ax.scatter([x_end], [y_end], color=color, s=30)


def draw_user_link(ax, tx_x, tx_y, user_x, user_y, label, color):
    ax.plot(
        [tx_x, user_x],
        [tx_y, user_y],
        linestyle="--",
        linewidth=1.5,
        color=color,
        label=label,
    )

    ax.scatter([user_x], [user_y], color=color, s=60)


def format_info_text(row):
    lines = [
        f"sample_id: {int(row['sample_id'])}",
        "",
        "User 1",
        f"  position: ({row['u1_x']:.2f}, {row['u1_y']:.2f}) m",
        f"  user angle: {row['u1_angle_deg']:.2f} deg",
        f"  target angle: {row['tx0_target_angle_deg']:.2f} deg",
        f"  sector: {int(row['tx0_sector_label'])}",
        f"  codebook: {int(row['tx0_codebook_label'])}",
        "",
        "User 2",
        f"  position: ({row['u2_x']:.2f}, {row['u2_y']:.2f}) m",
        f"  user angle: {row['u2_angle_deg']:.2f} deg",
        f"  target angle: {row['tx1_target_angle_deg']:.2f} deg",
        f"  sector: {int(row['tx1_sector_label'])}",
        f"  codebook: {int(row['tx1_codebook_label'])}",
    ]

    extra_fields = [
        ("target_sinr_u1_db", "U1 SINR", "dB"),
        ("target_sinr_u2_db", "U2 SINR", "dB"),
        ("target_rate_u1_bpshz", "U1 rate", "b/s/Hz"),
        ("target_rate_u2_bpshz", "U2 rate", "b/s/Hz"),
        ("target_sum_rate_bpshz", "Sum rate", "b/s/Hz"),
    ]

    present = False
    for key, _, _ in extra_fields:
        if key in row.index:
            present = True
            break

    if present:
        lines.append("")
        for key, name, unit in extra_fields:
            if key in row.index:
                lines.append(f"{name}: {row[key]:.2f} {unit}")

    return "\n".join(lines)


def select_row(df):
    if USE_RANDOM_SAMPLE:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = int(rng.integers(0, len(df)))
        return df.iloc[idx]

    matches = df[df["sample_id"] == SAMPLE_ID]

    if len(matches) == 0:
        raise ValueError(f"SAMPLE_ID={SAMPLE_ID} not found in CSV")

    return matches.iloc[0]


def main():
    df = pd.read_csv(CSV_PATH)
    row = select_row(df)

    tx_x, tx_y = load_tx_position()

    u1_x = float(row["u1_x"])
    u1_y = float(row["u1_y"])
    u2_x = float(row["u2_x"])
    u2_y = float(row["u2_y"])

    d1 = np.sqrt((u1_x - tx_x) ** 2 + (u1_y - tx_y) ** 2)
    d2 = np.sqrt((u2_x - tx_x) ** 2 + (u2_y - tx_y) ** 2)
    beam_length = BEAM_SCALE_MARGIN * max(d1, d2)

    plt.figure(figsize=(9, 8))
    ax = plt.gca()

    ax.scatter([tx_x], [tx_y], s=120, marker="s", label="TX")

    draw_user_link(ax, tx_x, tx_y, u1_x, u1_y, "TX to User 1", "tab:blue")
    draw_user_link(ax, tx_x, tx_y, u2_x, u2_y, "TX to User 2", "tab:orange")

    draw_beam(
        ax,
        tx_x,
        tx_y,
        float(row["tx0_target_angle_deg"]),
        beam_length,
        f"TX0 beam: sector {int(row['tx0_sector_label'])}, codebook {int(row['tx0_codebook_label'])}",
        "tab:green",
    )

    draw_beam(
        ax,
        tx_x,
        tx_y,
        float(row["tx1_target_angle_deg"]),
        beam_length,
        f"TX1 beam: sector {int(row['tx1_sector_label'])}, codebook {int(row['tx1_codebook_label'])}",
        "tab:red",
    )

    ax.text(
        u1_x,
        u1_y,
        f" U1\n angle={row['u1_angle_deg']:.1f}°",
        va="bottom",
    )

    ax.text(
        u2_x,
        u2_y,
        f" U2\n angle={row['u2_angle_deg']:.1f}°",
        va="bottom",
    )

    info_text = format_info_text(row)

    ax.text(
        1.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.set_xlabel("x-position [m]")
    ax.set_ylabel("y-position [m]")
    ax.set_title("RayNet MU-MIMO Dataset Sample Visualizer")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="upper left")

    x_vals = [tx_x, u1_x, u2_x]
    y_vals = [tx_y, u1_y, u2_y]

    x_min = min(x_vals) - 1.0
    x_max = max(x_vals) + 1.0
    y_min = min(y_vals) - 1.0
    y_max = max(y_vals) + 1.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
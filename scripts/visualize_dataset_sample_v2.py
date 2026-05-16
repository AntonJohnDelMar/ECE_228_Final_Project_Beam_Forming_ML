import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "outputs/dataset_v3.csv"
METADATA_PATH = "outputs/dataset_v3_metadata.json"

SAMPLE_ID = 0
USE_RANDOM_SAMPLE = False
RANDOM_SEED = 7

BEAM_SCALE_MARGIN = 1.15


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def select_row(df):
    if USE_RANDOM_SAMPLE:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = int(rng.integers(0, len(df)))
        return df.iloc[idx]

    matches = df[df["sample_id"] == SAMPLE_ID]

    if len(matches) == 0:
        raise ValueError(f"SAMPLE_ID={SAMPLE_ID} not found in CSV")

    return matches.iloc[0]


def endpoint_from_angle(tx_x, tx_y, angle_deg, length):
    angle_rad = np.deg2rad(angle_deg)

    x_end = tx_x + length * np.cos(angle_rad)
    y_end = tx_y + length * np.sin(angle_rad)

    return x_end, y_end


def draw_user_link(ax, tx_x, tx_y, user_x, user_y, label, color):
    ax.plot(
        [tx_x, user_x],
        [tx_y, user_y],
        linestyle="--",
        linewidth=1.5,
        color=color,
        alpha=0.8,
        label=label,
    )


def draw_beam(ax, tx_x, tx_y, angle_deg, length, label, color):
    x_end, y_end = endpoint_from_angle(tx_x, tx_y, angle_deg, length)

    ax.plot(
        [tx_x, x_end],
        [tx_y, y_end],
        linewidth=2.7,
        color=color,
        label=label,
    )

    ax.scatter([x_end], [y_end], color=color, s=35)


def format_info_text(row):
    lines = [
        f"sample_id: {int(row['sample_id'])}",
        "",
        "TX0 -> User 1",
        f"  sector: {int(row['tx0_sector_label'])}",
        f"  codebook: {int(row['tx0_codebook_label'])}",
        f"  target angle: {row['tx0_target_angle_deg']:.2f} deg",
        f"  U1 angle: {row['u1_angle_deg']:.2f} deg",
        "",
        "TX1 -> User 2",
        f"  sector: {int(row['tx1_sector_label'])}",
        f"  codebook: {int(row['tx1_codebook_label'])}",
        f"  target angle: {row['tx1_target_angle_deg']:.2f} deg",
        f"  U2 angle: {row['u2_angle_deg']:.2f} deg",
    ]

    optional_fields = [
        ("target_sinr_u1_db", "U1 SINR", "dB"),
        ("target_sinr_u2_db", "U2 SINR", "dB"),
        ("target_rate_u1_bpshz", "U1 rate", "b/s/Hz"),
        ("target_rate_u2_bpshz", "U2 rate", "b/s/Hz"),
        ("target_sum_rate_bpshz", "Sum rate", "b/s/Hz"),
        ("target_score", "Target score", ""),
        ("best_score", "Best score", ""),
        ("target_is_best", "Target is best", ""),
    ]

    present_fields = [entry for entry in optional_fields if entry[0] in row.index]

    if present_fields:
        lines.append("")
        lines.append("Metrics")

        for key, name, unit in present_fields:
            value = row[key]

            if key == "target_is_best":
                lines.append(f"  {name}: {int(value)}")
            elif unit:
                lines.append(f"  {name}: {value:.2f} {unit}")
            else:
                lines.append(f"  {name}: {value:.2f}")

    link_fields = [
        ("h11_power_db", "h11 TX0->U1"),
        ("h12_power_db", "h12 TX0->U2"),
        ("h21_power_db", "h21 TX1->U1"),
        ("h22_power_db", "h22 TX1->U2"),
    ]

    present_links = [entry for entry in link_fields if entry[0] in row.index]

    if present_links:
        lines.append("")
        lines.append("Link power")

        for key, name in present_links:
            lines.append(f"  {name}: {row[key]:.2f} dB")

    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV_PATH)
    metadata = load_metadata()
    row = select_row(df)

    tx0_x = float(row["tx0_x"])
    tx0_y = float(row["tx0_y"])
    tx1_x = float(row["tx1_x"])
    tx1_y = float(row["tx1_y"])

    u1_x = float(row["u1_x"])
    u1_y = float(row["u1_y"])
    u2_x = float(row["u2_x"])
    u2_y = float(row["u2_y"])

    dists = [
        np.hypot(u1_x - tx0_x, u1_y - tx0_y),
        np.hypot(u2_x - tx1_x, u2_y - tx1_y),
        np.hypot(u1_x - tx1_x, u1_y - tx1_y),
        np.hypot(u2_x - tx0_x, u2_y - tx0_y),
    ]

    beam_length = BEAM_SCALE_MARGIN * max(dists)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter([tx0_x], [tx0_y], s=140, marker="s", label="TX0", color="tab:green")
    ax.scatter([tx1_x], [tx1_y], s=140, marker="s", label="TX1", color="tab:red")

    ax.scatter([u1_x], [u1_y], s=90, marker="o", label="User 1", color="tab:blue")
    ax.scatter([u2_x], [u2_y], s=90, marker="o", label="User 2", color="tab:orange")

    draw_user_link(
        ax,
        tx0_x,
        tx0_y,
        u1_x,
        u1_y,
        "desired: TX0 -> U1",
        "tab:blue",
    )

    draw_user_link(
        ax,
        tx1_x,
        tx1_y,
        u2_x,
        u2_y,
        "desired: TX1 -> U2",
        "tab:orange",
    )

    draw_user_link(
        ax,
        tx1_x,
        tx1_y,
        u1_x,
        u1_y,
        "interference: TX1 -> U1",
        "gray",
    )

    draw_user_link(
        ax,
        tx0_x,
        tx0_y,
        u2_x,
        u2_y,
        "interference: TX0 -> U2",
        "silver",
    )

    draw_beam(
        ax,
        tx0_x,
        tx0_y,
        float(row["tx0_target_angle_deg"]),
        beam_length,
        f"TX0 beam: sector {int(row['tx0_sector_label'])}, codebook {int(row['tx0_codebook_label'])}",
        "tab:green",
    )

    draw_beam(
        ax,
        tx1_x,
        tx1_y,
        float(row["tx1_target_angle_deg"]),
        beam_length,
        f"TX1 beam: sector {int(row['tx1_sector_label'])}, codebook {int(row['tx1_codebook_label'])}",
        "tab:red",
    )

    ax.text(
        tx0_x,
        tx0_y,
        " TX0",
        va="bottom",
        ha="left",
        fontsize=10,
    )

    ax.text(
        tx1_x,
        tx1_y,
        " TX1",
        va="bottom",
        ha="left",
        fontsize=10,
    )

    ax.text(
        u1_x,
        u1_y,
        f" U1\n angle={row['u1_angle_deg']:.1f}°",
        va="bottom",
        ha="left",
        fontsize=10,
    )

    ax.text(
        u2_x,
        u2_y,
        f" U2\n angle={row['u2_angle_deg']:.1f}°",
        va="bottom",
        ha="left",
        fontsize=10,
    )

    info_text = format_info_text(row)

    ax.text(
        1.03,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )

    ax.set_xlabel("x-position [m]")
    ax.set_ylabel("y-position [m]")
    ax.set_title("V3 Two-TX RayNet MU-MIMO Dataset Sample")

    ax.axis("equal")
    ax.grid(True)

    all_x = [tx0_x, tx1_x, u1_x, u2_x]
    all_y = [tx0_y, tx1_y, u1_y, u2_y]

    x_pad = max(1.0, 0.15 * (max(all_x) - min(all_x) + 1e-9))
    y_pad = max(1.0, 0.15 * (max(all_y) - min(all_y) + 1e-9))

    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
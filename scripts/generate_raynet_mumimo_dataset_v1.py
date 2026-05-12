import json
import os

import mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")

import numpy as np
import pandas as pd

import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver


C0 = 299_792_458.0


CONFIG = {
    "num_samples": 50,
    "seed": 7,

    "frequency_hz": 60e9,

    "tx_position": [0.0, 0.0, 1.5],
    "rx_height_m": 1.5,

    "user_distance_min_m": 2.0,
    "user_distance_max_m": 10.0,
    "user_angle_min_deg": -55.0,
    "user_angle_max_deg": 55.0,
    "min_user_angular_separation_deg": 8.0,

    "num_beams": 8,
    "num_codebooks": 32,

    "beam_angle_min_deg": -52.5,
    "beam_angle_max_deg": 52.5,
    "codebook_refinement_min_deg": -7.0,
    "codebook_refinement_max_deg": 7.0,

    "beam_shape_exponent": 12,
    "beam_gain_floor_db": -30.0,

    "noise_power_linear": 1e-10,
    "interference_factor": 1.0,

    "required_rate_min_bpshz": 1.0,
    "required_rate_max_bpshz": 5.0,
    "qos_penalty_weight": 2.0,

    "output_csv": "outputs/raynet_mumimo_dataset_v1.csv",
    "output_npz": "outputs/raynet_mumimo_dataset_v1.npz",
    "output_metadata": "outputs/raynet_mumimo_dataset_v1_metadata.json",
}


def setup_scene():
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.frequency = CONFIG["frequency_hz"]

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V",
    )

    tx = Transmitter(
        name="tx",
        position=CONFIG["tx_position"],
    )

    scene.add(tx)

    return scene, tx


def sample_user_position(rng):
    distance = rng.uniform(
        CONFIG["user_distance_min_m"],
        CONFIG["user_distance_max_m"],
    )

    angle_deg = rng.uniform(
        CONFIG["user_angle_min_deg"],
        CONFIG["user_angle_max_deg"],
    )

    angle_rad = np.deg2rad(angle_deg)

    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    z = CONFIG["rx_height_m"]

    return np.array([x, y, z], dtype=float), distance, angle_deg


def sample_two_users(rng):
    while True:
        u1_pos, u1_dist, u1_angle = sample_user_position(rng)
        u2_pos, u2_dist, u2_angle = sample_user_position(rng)

        angular_sep = abs(u1_angle - u2_angle)

        if angular_sep >= CONFIG["min_user_angular_separation_deg"]:
            return {
                "u1_pos": u1_pos,
                "u1_distance_m": u1_dist,
                "u1_angle_deg": u1_angle,
                "u2_pos": u2_pos,
                "u2_distance_m": u2_dist,
                "u2_angle_deg": u2_angle,
                "angular_separation_deg": angular_sep,
            }


def trace_single_user(scene, tx, p_solver, rx_name, rx_position, seed):
    rx = Receiver(
        name=rx_name,
        position=rx_position.tolist(),
    )

    scene.add(rx)
    tx.look_at(rx)

    paths = p_solver(
        scene=scene,
        max_depth=1,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=True,
        seed=seed,
    )

    a, tau = paths.cir(normalize_delays=False, out_type="numpy")

    scene.remove(rx_name)

    return a, tau


def extract_channel_features(a, tau):
    a_flat = np.asarray(a).flatten()
    tau_flat = np.asarray(tau).flatten()

    mags = np.abs(a_flat)
    valid = np.isfinite(mags) & np.isfinite(tau_flat)

    mags = mags[valid]
    tau_flat = tau_flat[valid]

    if len(mags) == 0:
        return {
            "num_paths": 0,
            "channel_power_linear": 0.0,
            "channel_power_db": -300.0,
            "strongest_path_mag": 0.0,
            "strongest_delay_ns": np.nan,
            "rms_delay_spread_ns": np.nan,
        }

    powers = mags**2
    total_power = np.sum(powers)
    total_power_db = 10.0 * np.log10(total_power + 1e-30)

    strongest_idx = int(np.argmax(mags))
    strongest_path_mag = float(mags[strongest_idx])
    strongest_delay_ns = float(tau_flat[strongest_idx] / 1e-9)

    mean_delay = np.sum(powers * tau_flat) / (total_power + 1e-30)
    rms_delay = np.sqrt(np.sum(powers * (tau_flat - mean_delay) ** 2) / (total_power + 1e-30))
    rms_delay_spread_ns = float(rms_delay / 1e-9)

    return {
        "num_paths": int(len(mags)),
        "channel_power_linear": float(total_power),
        "channel_power_db": float(total_power_db),
        "strongest_path_mag": strongest_path_mag,
        "strongest_delay_ns": strongest_delay_ns,
        "rms_delay_spread_ns": rms_delay_spread_ns,
    }


def make_candidate_configs():
    beam_angles = np.linspace(
        CONFIG["beam_angle_min_deg"],
        CONFIG["beam_angle_max_deg"],
        CONFIG["num_beams"],
    )

    codebook_offsets = np.linspace(
        CONFIG["codebook_refinement_min_deg"],
        CONFIG["codebook_refinement_max_deg"],
        CONFIG["num_codebooks"],
    )

    candidates = []

    for beam_idx, beam_angle in enumerate(beam_angles):
        for codebook_idx, offset in enumerate(codebook_offsets):
            candidates.append({
                "beam_idx": beam_idx,
                "codebook_idx": codebook_idx,
                "beam_angle_deg": beam_angle,
                "codebook_offset_deg": offset,
                "effective_angle_deg": beam_angle + offset,
            })

    return pd.DataFrame(candidates)


def beam_gain_linear(user_angle_deg, candidate_angles_deg):
    angle_error = user_angle_deg - candidate_angles_deg

    gain = np.cos(np.deg2rad(angle_error))
    gain = np.maximum(gain, 0.0)
    gain = gain ** CONFIG["beam_shape_exponent"]

    floor_linear = 10.0 ** (CONFIG["beam_gain_floor_db"] / 10.0)
    gain = np.maximum(gain, floor_linear)

    return gain


def evaluate_joint_configurations(u1, u2, candidates):
    candidate_angles = candidates["effective_angle_deg"].to_numpy()

    gain_u1 = beam_gain_linear(u1["angle_deg"], candidate_angles)
    gain_u2 = beam_gain_linear(u2["angle_deg"], candidate_angles)

    h1 = u1["channel_power_linear"]
    h2 = u2["channel_power_linear"]

    noise = CONFIG["noise_power_linear"]
    alpha = CONFIG["interference_factor"]

    desired_u1 = h1 * gain_u1[:, None]
    interferer_u1 = alpha * h1 * gain_u1[None, :]

    desired_u2 = h2 * gain_u2[None, :]
    interferer_u2 = alpha * h2 * gain_u2[:, None]

    sinr_u1 = desired_u1 / (interferer_u1 + noise)
    sinr_u2 = desired_u2 / (interferer_u2 + noise)

    rate_u1 = np.log2(1.0 + sinr_u1)
    rate_u2 = np.log2(1.0 + sinr_u2)

    req_u1 = u1["required_rate_bpshz"]
    req_u2 = u2["required_rate_bpshz"]

    deficit_u1 = np.maximum(0.0, req_u1 - rate_u1)
    deficit_u2 = np.maximum(0.0, req_u2 - rate_u2)

    score = (
        rate_u1
        + rate_u2
        - CONFIG["qos_penalty_weight"] * (deficit_u1**2 + deficit_u2**2)
    )

    best_flat_idx = int(np.argmax(score))
    best_tx0_idx, best_tx1_idx = np.unravel_index(best_flat_idx, score.shape)

    best_tx0 = candidates.iloc[best_tx0_idx]
    best_tx1 = candidates.iloc[best_tx1_idx]

    independent_u1_idx = int(np.argmax(gain_u1))
    independent_u2_idx = int(np.argmax(gain_u2))

    return {
        "tx0_beam_label": int(best_tx0["beam_idx"]),
        "tx0_codebook_label": int(best_tx0["codebook_idx"]),
        "tx0_effective_angle_deg": float(best_tx0["effective_angle_deg"]),

        "tx1_beam_label": int(best_tx1["beam_idx"]),
        "tx1_codebook_label": int(best_tx1["codebook_idx"]),
        "tx1_effective_angle_deg": float(best_tx1["effective_angle_deg"]),

        "independent_u1_beam": int(candidates.iloc[independent_u1_idx]["beam_idx"]),
        "independent_u1_codebook": int(candidates.iloc[independent_u1_idx]["codebook_idx"]),
        "independent_u2_beam": int(candidates.iloc[independent_u2_idx]["beam_idx"]),
        "independent_u2_codebook": int(candidates.iloc[independent_u2_idx]["codebook_idx"]),

        "u1_sinr_db": float(10.0 * np.log10(sinr_u1[best_tx0_idx, best_tx1_idx] + 1e-30)),
        "u2_sinr_db": float(10.0 * np.log10(sinr_u2[best_tx0_idx, best_tx1_idx] + 1e-30)),
        "u1_rate_bpshz": float(rate_u1[best_tx0_idx, best_tx1_idx]),
        "u2_rate_bpshz": float(rate_u2[best_tx0_idx, best_tx1_idx]),
        "sum_rate_bpshz": float(rate_u1[best_tx0_idx, best_tx1_idx] + rate_u2[best_tx0_idx, best_tx1_idx]),
        "joint_score": float(score[best_tx0_idx, best_tx1_idx]),
    }


def make_model_arrays(df):
    feature_cols = [
        "u1_distance_m",
        "u1_angle_deg",
        "u1_required_rate_bpshz",
        "u1_pilot_snr_db",
        "u2_distance_m",
        "u2_angle_deg",
        "u2_required_rate_bpshz",
        "u2_pilot_snr_db",
    ]

    beam_label_cols = [
        "tx0_beam_label",
        "tx1_beam_label",
    ]

    codebook_label_cols = [
        "tx0_codebook_label",
        "tx1_codebook_label",
    ]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_beam = df[beam_label_cols].to_numpy(dtype=np.int64)
    y_codebook = df[codebook_label_cols].to_numpy(dtype=np.int64)

    return X, y_beam, y_codebook, feature_cols


def main():
    os.makedirs("outputs", exist_ok=True)

    rng = np.random.default_rng(CONFIG["seed"])

    scene, tx = setup_scene()
    p_solver = PathSolver()
    candidates = make_candidate_configs()

    rows = []

    for sample_id in range(CONFIG["num_samples"]):
        users = sample_two_users(rng)

        u1_required_rate = rng.uniform(
            CONFIG["required_rate_min_bpshz"],
            CONFIG["required_rate_max_bpshz"],
        )

        u2_required_rate = rng.uniform(
            CONFIG["required_rate_min_bpshz"],
            CONFIG["required_rate_max_bpshz"],
        )

        a1, tau1 = trace_single_user(
            scene=scene,
            tx=tx,
            p_solver=p_solver,
            rx_name="rx_u1",
            rx_position=users["u1_pos"],
            seed=10_000 + sample_id,
        )

        a2, tau2 = trace_single_user(
            scene=scene,
            tx=tx,
            p_solver=p_solver,
            rx_name="rx_u2",
            rx_position=users["u2_pos"],
            seed=20_000 + sample_id,
        )

        u1_ch = extract_channel_features(a1, tau1)
        u2_ch = extract_channel_features(a2, tau2)

        u1_pilot_snr_linear = u1_ch["channel_power_linear"] / CONFIG["noise_power_linear"]
        u2_pilot_snr_linear = u2_ch["channel_power_linear"] / CONFIG["noise_power_linear"]

        u1_pilot_snr_db = 10.0 * np.log10(u1_pilot_snr_linear + 1e-30)
        u2_pilot_snr_db = 10.0 * np.log10(u2_pilot_snr_linear + 1e-30)

        u1_for_scoring = {
            "angle_deg": users["u1_angle_deg"],
            "channel_power_linear": u1_ch["channel_power_linear"],
            "required_rate_bpshz": u1_required_rate,
        }

        u2_for_scoring = {
            "angle_deg": users["u2_angle_deg"],
            "channel_power_linear": u2_ch["channel_power_linear"],
            "required_rate_bpshz": u2_required_rate,
        }

        labels = evaluate_joint_configurations(
            u1=u1_for_scoring,
            u2=u2_for_scoring,
            candidates=candidates,
        )

        row = {
            "sample_id": sample_id,

            "u1_x": users["u1_pos"][0],
            "u1_y": users["u1_pos"][1],
            "u1_z": users["u1_pos"][2],
            "u1_distance_m": users["u1_distance_m"],
            "u1_angle_deg": users["u1_angle_deg"],
            "u1_required_rate_bpshz": u1_required_rate,
            "u1_pilot_snr_db": u1_pilot_snr_db,

            "u2_x": users["u2_pos"][0],
            "u2_y": users["u2_pos"][1],
            "u2_z": users["u2_pos"][2],
            "u2_distance_m": users["u2_distance_m"],
            "u2_angle_deg": users["u2_angle_deg"],
            "u2_required_rate_bpshz": u2_required_rate,
            "u2_pilot_snr_db": u2_pilot_snr_db,

            "angular_separation_deg": users["angular_separation_deg"],

            "u1_num_paths": u1_ch["num_paths"],
            "u1_channel_power_db": u1_ch["channel_power_db"],
            "u1_strongest_delay_ns": u1_ch["strongest_delay_ns"],
            "u1_rms_delay_spread_ns": u1_ch["rms_delay_spread_ns"],

            "u2_num_paths": u2_ch["num_paths"],
            "u2_channel_power_db": u2_ch["channel_power_db"],
            "u2_strongest_delay_ns": u2_ch["strongest_delay_ns"],
            "u2_rms_delay_spread_ns": u2_ch["rms_delay_spread_ns"],

            **labels,
        }

        rows.append(row)

        print(
            f"sample {sample_id:04d}: "
            f"u1 angle={users['u1_angle_deg']:+6.2f} deg, "
            f"u2 angle={users['u2_angle_deg']:+6.2f} deg, "
            f"labels=({labels['tx0_beam_label']}, {labels['tx0_codebook_label']}), "
            f"({labels['tx1_beam_label']}, {labels['tx1_codebook_label']}), "
            f"sum_rate={labels['sum_rate_bpshz']:.2f} b/s/Hz"
        )

    df = pd.DataFrame(rows)

    X, y_beam, y_codebook, feature_cols = make_model_arrays(df)

    df.to_csv(CONFIG["output_csv"], index=False)

    np.savez(
        CONFIG["output_npz"],
        X=X,
        y_beam=y_beam,
        y_codebook=y_codebook,
    )

    metadata = {
        "description": "RayNet-like two-user MU-MIMO analog beam/codebook selection dataset.",
        "feature_columns": feature_cols,
        "beam_label_columns": ["tx0_beam_label", "tx1_beam_label"],
        "codebook_label_columns": ["tx0_codebook_label", "tx1_codebook_label"],
        "config": CONFIG,
        "note": (
            "This V1 dataset uses Sionna RT for channel/path generation and an "
            "analytical beam/codebook gain model for labels. Replace the analytical "
            "gain model with measured/simulated RayNet beam patterns in V2."
        ),
    }

    with open(CONFIG["output_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(f"  {CONFIG['output_csv']}")
    print(f"  {CONFIG['output_npz']}")
    print(f"  {CONFIG['output_metadata']}")
    print("\nFeature matrix shape:", X.shape)
    print("Beam labels shape:", y_beam.shape)
    print("Codebook labels shape:", y_codebook.shape)


if __name__ == "__main__":
    main()
import mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")

import os
import numpy as np
import pandas as pd

import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver


C0 = 299_792_458.0


def setup_scene():
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.frequency = 60e9

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
        position=[-2.0, 0.0, 1.5],
    )

    scene.add(tx)

    return scene, tx


def make_rx_grid(x_min=0.5, x_max=4.0, y_min=-2.0, y_max=2.0, z=1.5, nx=8, ny=8):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)

    positions = []

    for x in xs:
        for y in ys:
            positions.append([x, y, z])

    return np.array(positions, dtype=float)


def compute_geometry_features(tx_pos, rx_pos):
    delta = rx_pos - tx_pos
    distance = np.linalg.norm(delta)

    azimuth_rad = np.arctan2(delta[1], delta[0])
    azimuth_deg = np.rad2deg(azimuth_rad)

    horizontal_distance = np.linalg.norm(delta[:2])
    elevation_rad = np.arctan2(delta[2], horizontal_distance)
    elevation_deg = np.rad2deg(elevation_rad)

    return distance, azimuth_deg, elevation_deg


def extract_path_features(a, tau):
    a_flat = np.asarray(a).flatten()
    tau_flat = np.asarray(tau).flatten()

    mags = np.abs(a_flat)

    valid = np.isfinite(mags) & np.isfinite(tau_flat)

    mags = mags[valid]
    tau_flat = tau_flat[valid]

    if len(mags) == 0:
        return {
            "num_paths": 0,
            "strongest_delay_ns": np.nan,
            "strongest_mag": 0.0,
            "strongest_power_db": -300.0,
            "total_power": 0.0,
            "total_power_db": -300.0,
            "rms_delay_spread_ns": np.nan,
        }

    powers = mags**2
    total_power = np.sum(powers)
    total_power_db = 10.0 * np.log10(total_power + 1e-30)

    strongest_idx = np.argmax(mags)
    strongest_delay_ns = tau_flat[strongest_idx] / 1e-9
    strongest_mag = mags[strongest_idx]
    strongest_power_db = 20.0 * np.log10(strongest_mag + 1e-30)

    tau_mean = np.sum(powers * tau_flat) / (total_power + 1e-30)
    tau_rms = np.sqrt(np.sum(powers * (tau_flat - tau_mean)**2) / (total_power + 1e-30))
    rms_delay_spread_ns = tau_rms / 1e-9

    return {
        "num_paths": len(mags),
        "strongest_delay_ns": strongest_delay_ns,
        "strongest_mag": strongest_mag,
        "strongest_power_db": strongest_power_db,
        "total_power": total_power,
        "total_power_db": total_power_db,
        "rms_delay_spread_ns": rms_delay_spread_ns,
    }


def main():
    os.makedirs("outputs", exist_ok=True)

    scene, tx = setup_scene()
    p_solver = PathSolver()

    tx_pos = np.array(tx.position, dtype=float)
    rx_positions = make_rx_grid(nx=8, ny=8)

    rows = []

    for sample_id, rx_pos in enumerate(rx_positions):
        rx_name = f"rx_{sample_id}"

        rx = Receiver(
            name=rx_name,
            position=rx_pos.tolist(),
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
            seed=sample_id,
        )

        a, tau = paths.cir(normalize_delays=False, out_type="numpy")

        distance, azimuth_deg, elevation_deg = compute_geometry_features(tx_pos, rx_pos)
        path_features = extract_path_features(a, tau)

        row = {
            "sample_id": sample_id,
            "tx_x": tx_pos[0],
            "tx_y": tx_pos[1],
            "tx_z": tx_pos[2],
            "rx_x": rx_pos[0],
            "rx_y": rx_pos[1],
            "rx_z": rx_pos[2],
            "distance_m": distance,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "frequency_hz": scene.frequency,
            **path_features,
        }

        rows.append(row)

        print(
            f"sample {sample_id:03d}: "
            f"rx=({rx_pos[0]:.2f}, {rx_pos[1]:.2f}, {rx_pos[2]:.2f}), "
            f"paths={path_features['num_paths']}, "
            f"power={path_features['total_power_db']:.2f} dB"
        )

        scene.remove(rx_name)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/sionna_rt_dataset_v1.csv", index=False)

    print("\nSaved outputs/sionna_rt_dataset_v1.csv")
    print(df.head())


if __name__ == "__main__":
    main()
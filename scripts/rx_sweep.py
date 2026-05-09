import mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")

import numpy as np
import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver


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
    position=[-2, 0, 1.5],
)

scene.add(tx)

p_solver = PathSolver()

rx_positions = np.array([
    [1.0, -1.0, 1.5],
    [1.5, -0.5, 1.5],
    [2.0,  0.0, 1.5],
    [2.5,  0.5, 1.5],
    [3.0,  1.0, 1.5],
])

all_tau = []
all_mag = []

for idx, rx_pos in enumerate(rx_positions):
    rx_name = f"rx_{idx}"

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
        seed=1,
    )

    a, tau = paths.cir(normalize_delays=False, out_type="numpy")

    mag = np.abs(a).flatten()
    delays_ns = tau.flatten() / 1e-9

    print(f"\nRX {idx}: position = {rx_pos}")
    print("Delays [ns]:", delays_ns)
    print("Magnitudes:", mag)

    all_tau.append(tau)
    all_mag.append(mag)

    scene.remove(rx_name)

all_tau_obj = np.empty(len(all_tau), dtype=object)
all_mag_obj = np.empty(len(all_mag), dtype=object)

for i in range(len(all_tau)):
    all_tau_obj[i] = all_tau[i]
    all_mag_obj[i] = all_mag[i]

np.savez(
    "outputs/rx_sweep_result.npz",
    rx_positions=rx_positions,
    all_tau=all_tau_obj,
    all_mag=all_mag_obj,
    frequency=scene.frequency,
)

print("\nSaved outputs/rx_sweep_result.npz")
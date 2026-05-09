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

rx = Receiver(
    name="rx",
    position=[2, 0, 1.5],
)

scene.add(tx)
scene.add(rx)

tx.look_at(rx)

p_solver = PathSolver()

paths = p_solver(
    scene=scene,
    max_depth=0,
    los=True,
    specular_reflection=False,
    diffuse_reflection=False,
    refraction=False,
    synthetic_array=True,
    seed=1,
)

a, tau = paths.cir(normalize_delays=False, out_type="numpy")

print("CIR coefficient shape:", a.shape)
print("Delay shape:", tau.shape)
print("Path delays [ns]:")
print(tau.flatten() / 1e-9)

print("Path magnitudes:")
print(np.abs(a).flatten())

np.savez(
    "outputs/first_link_result.npz",
    a=a,
    tau=tau,
    tx_position=np.array(tx.position),
    rx_position=np.array(rx.position),
    frequency=scene.frequency,
)
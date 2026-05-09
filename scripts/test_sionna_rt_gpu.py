import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import sionna.rt
from sionna.rt import load_scene

print("Sionna RT import OK")
print("Using Mitsuba variant:", mi.variant())

scene = load_scene(sionna.rt.scene.simple_reflector)
print("Loaded simple_reflector scene OK")
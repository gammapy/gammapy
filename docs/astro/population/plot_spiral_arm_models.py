"""Plot Milky Way spiral arm models."""
import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Quantity
from gammapy.astro.population.spatial import ValleeSpiral, FaucherSpiral

fig = plt.figure(figsize=(7, 8))
rect = [0.12, 0.12, 0.85, 0.85]
ax_cartesian = fig.add_axes(rect)
ax_cartesian.set_aspect("equal")

vallee_spiral = ValleeSpiral()
faucher_spiral = FaucherSpiral()

radius = Quantity(np.arange(2.1, 10, 0.1), "kpc")

for spiralarm_index in range(4):
    # TODO: spiral arm index is different for the two spirals
    # -> spirals by the same name have different colors.
    # Should we change this in the implementation or here in plotting?
    color = "C{}".format(spiralarm_index)

    # Plot Vallee spiral
    x, y = vallee_spiral.xy_position(radius=radius, spiralarm_index=spiralarm_index)
    name = vallee_spiral.spiralarms[spiralarm_index]
    ax_cartesian.plot(x, y, label="Vallee " + name, ls="-", color=color)

    # Plot Faucher spiral
    x, y = faucher_spiral.xy_position(radius=radius, spiralarm_index=spiralarm_index)
    name = faucher_spiral.spiralarms[spiralarm_index]
    ax_cartesian.plot(x, y, label="Faucher " + name, ls="-.", color=color)

ax_cartesian.plot(vallee_spiral.bar["x"], vallee_spiral.bar["y"])

ax_cartesian.set_xlabel("x (kpc)")
ax_cartesian.set_ylabel("y (kpc)")
ax_cartesian.set_xlim(-12, 12)
ax_cartesian.set_ylim(-15, 12)
ax_cartesian.legend(ncol=2, loc="lower right")

plt.grid()
plt.show()

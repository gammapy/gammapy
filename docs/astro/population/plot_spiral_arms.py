"""Plot Milky Way spiral arms."""
import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Quantity
from gammapy.astro.population import simulate
from gammapy.astro.population import FaucherSpiral
from gammapy.utils.coordinates import polar, cartesian

catalog = simulate.make_base_catalog_galactic(
    n_sources=int(1e4), rad_dis="YK04", vel_dis="H05", max_age=Quantity(1e6, "yr")
)

spiral = FaucherSpiral()

fig = plt.figure(figsize=(6, 6))
rect = [0.12, 0.12, 0.85, 0.85]
ax_cartesian = fig.add_axes(rect)
ax_cartesian.set_aspect("equal")

ax_polar = fig.add_axes(rect, polar=True, frameon=False)
ax_polar.axes.get_xaxis().set_ticklabels([])
ax_polar.axes.get_yaxis().set_ticklabels([])

ax_cartesian.plot(
    catalog["x"],
    catalog["y"],
    marker=".",
    linestyle="none",
    markersize=5,
    alpha=0.3,
    fillstyle="full",
)
ax_cartesian.set_xlim(-20, 20)
ax_cartesian.set_ylim(-20, 20)
ax_cartesian.set_xlabel("x [kpc]", labelpad=2)
ax_cartesian.set_ylabel("y [kpc]", labelpad=-4)
ax_cartesian.plot(
    0, 8, color="k", markersize=10, fillstyle="none", marker="*", linewidth=2
)
ax_cartesian.annotate(
    "Sun",
    xy=(0, 8),
    xycoords="data",
    xytext=(-15, 15),
    arrowprops=dict(arrowstyle="->", color="k"),
    weight=400,
)

plt.grid(True)

# TODO: document what these magic numbers are or simplify the code
# `other_idx = [95, 90, 80, 80]` and below `theta_idx = int(other_idx * 0.97)`
for spiral_idx, other_idx in zip(range(4), [95, 90, 80, 80]):
    spiralarm_name = spiral.spiralarms[spiral_idx]
    theta_0 = spiral.theta_0[spiral_idx].value

    theta = Quantity(np.linspace(theta_0, theta_0 + 2 * np.pi, 100), "rad")
    x, y = spiral.xy_position(theta=theta, spiralarm_index=spiral_idx)
    ax_cartesian.plot(x.value, y.value, color="k")
    rad, phi = polar(x[other_idx], y[other_idx])
    x_pos, y_pos = cartesian(rad + Quantity(1, "kpc"), phi)
    theta_idx = int(other_idx * 0.97)
    rotation = theta[theta_idx].to("deg").value
    ax_cartesian.text(
        x_pos.value,
        y_pos.value,
        spiralarm_name,
        ha="center",
        va="center",
        rotation=rotation - 90,
        weight=400,
    )
plt.show()

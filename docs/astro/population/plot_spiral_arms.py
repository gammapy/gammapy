import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Quantity
from gammapy.astro.population import simulate
from gammapy.astro.population import FaucherSpiral
from gammapy.utils.coordinates import polar, cartesian

catalog = simulate.make_base_catalog_galactic(n_sources=1E4, rad_dis='YK04',
                                              vel_dis='H05', max_age=Quantity(1E6, 'yr'))

spiral = FaucherSpiral()

fig = plt.figure(figsize=(6, 6))
rect = [0.12, 0.12, 0.85, 0.85]
ax_cartesian  = fig.add_axes(rect)
ax_cartesian.set_aspect('equal')
ax_polar = fig.add_axes(rect, polar=True, frameon=False)
ax_polar.axes.get_xaxis().set_ticklabels([])
ax_polar.axes.get_yaxis().set_ticklabels([])

ax_cartesian.plot(catalog['x'], catalog['y'], marker='.',
                  linestyle='none', markersize=5, alpha=0.3, fillstyle='full')
ax_cartesian.set_xlim(-20, 20)
ax_cartesian.set_ylim(-20, 20)
ax_cartesian.set_xlabel('x [kpc]', labelpad=2)
ax_cartesian.set_ylabel('y [kpc]', labelpad=-4)
ax_cartesian.plot(0, 8, color='k', markersize=10, fillstyle='none', marker='*', linewidth=2)
ax_cartesian.annotate('Sun', xy=(0, 8),  xycoords='data',
                xytext=(-15, 15),  arrowprops=dict(arrowstyle="->", color='k'), weight=400)

plt.grid(True)
from astropy.units import Quantity

ind = [95, 90, 80, 80]

for i in range(4):
    theta_0 = spiral.theta_0[i].value
    theta = Quantity(np.linspace(theta_0, theta_0 + 2 * np.pi, 100), 'rad')
    x, y = spiral.xy_position(theta=theta, spiralarm_index=i)
    ax_cartesian.plot(x.value, y.value, color='k')
    rad, phi = polar(x[ind[i]], y[ind[i]])
    x_pos, y_pos = cartesian(rad + Quantity(1, 'kpc'), phi)
    rotation = theta[ind[i] * 0.97].to('deg').value
    ax_cartesian.text(x_pos.value, y_pos.value, spiral.spiralarms[i],
             ha='center', va='center', rotation=rotation - 90, weight=400)  
plt.show()

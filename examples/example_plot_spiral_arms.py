"""Plot Milky Way spiral arm models."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.units import Quantity
from gammapy.astro.population.spatial import ValleeSpiral, FaucherSpiral
from gammapy.utils.mpl_style import gammapy_mpl_style

plt.style.use(gammapy_mpl_style)
fig = plt.figure(figsize=(12, 12))
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

vallee_spiral = ValleeSpiral()
faucher_spiral = FaucherSpiral()

#theta = np.arange(0, 720)
radius = Quantity(np.arange(2.1, 20, 0.1), 'kpc')

for spiralarm_index in range(4):
    # Plot Vallee spiral
    x, y = vallee_spiral.xy_position(radius=radius, spiralarm_index=spiralarm_index)
    name = vallee_spiral.spiralarms[spiralarm_index]
    axes.plot(x, y, label=name)

    # Plot Faucher spiral
    x, y = faucher_spiral.xy_position(radius=radius, spiralarm_index=spiralarm_index)
    name = faucher_spiral.spiralarms[spiralarm_index]
    axes.plot(x, y, ls='-.', label='Faucher ' + name)

axes.plot(vallee_spiral.bar['x'], vallee_spiral.bar['y'])

axes.set_xlabel('x (kpc)')
axes.set_ylabel('y (kpc)')
axes.set_xlim(-10, 10)
axes.set_ylim(-10, 10)
axes.legend(ncol=2, loc='lower right')

filename = 'vallee_spiral.pdf'
print('Writing {0}'.format(filename))
fig.savefig(filename)

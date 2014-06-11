# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Plot Milky Way spiral arm models.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gammapy.astro.population.spatial import ValleeSpiral, FaucherSpiral

vallee_spiral = ValleeSpiral()
faucher_spiral = FaucherSpiral()

#theta = np.arange(0, 720)
radius = np.arange(2.1, 20, 0.1)

for spiralarm_index in range(4):
    # Plot Vallee spiral
    x, y = vallee_spiral.xy_position(radius=radius, spiralarm_index=spiralarm_index)
    name = vallee_spiral.spiralarms[spiralarm_index]
    plt.plot(x, y, label=name)

    # Plot Faucher spiral
    x, y = faucher_spiral.xy_position(radius=radius, spiralarm_index=spiralarm_index)
    name = faucher_spiral.spiralarms[spiralarm_index]
    plt.plot(x, y, ls='-.', label='Faucher ' + name)

plt.plot(vallee_spiral.bar['x'], vallee_spiral.bar['y'])

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.legend(ncol=2)

filename = 'valee_spiral.pdf'
print('Writing {0}'.format(filename))
plt.savefig(filename)

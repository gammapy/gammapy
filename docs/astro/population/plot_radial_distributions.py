"""Plot radial surface density distributions of Galactic sources."""
import matplotlib.pyplot as plt
import numpy as np
from gammapy.astro.population import radial_distributions
from gammapy.utils.distributions import normalize

max_radius = 20  # kpc
r = np.linspace(0, max_radius, 100)
colors = ['b', 'k', 'k', 'b', 'g', 'g']

for color, key in zip(colors, radial_distributions):
    model = radial_distributions[key]()
    if model.evolved:
        linestyle = '-'
    else:
        linestyle = '--'
    label = model.__class__.__name__
    plt.plot(r, normalize(model, 0, max_radius)(r), color=color, linestyle=linestyle, label=label)
plt.xlim(0, max_radius)
plt.ylim(0, 0.28)
plt.xlabel('Galactocentric Distance [kpc]')
plt.ylabel('Normalized Surface Density [kpc^-2]')
plt.legend(prop={'size': 10})
plt.show()

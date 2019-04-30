"""Plot radial surface density distributions of Galactic sources."""
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from gammapy.astro.population import radial_distributions
from gammapy.utils.distributions import normalize

radius = np.linspace(0, 20, 100) * u.kpc

for key in radial_distributions:
    model = radial_distributions[key]()
    if model.evolved:
        linestyle = "-"
    else:
        linestyle = "--"
    label = model.__class__.__name__
    x = radius.value
    y = normalize(model, 0, radius[-1].value)(radius.value)
    plt.plot(x, y, linestyle=linestyle, label=label)

plt.xlim(0, radius[-1].value)
plt.ylim(0, 0.26)
plt.xlabel("Galactocentric Distance [kpc]")
plt.ylabel("Normalized Surface Density [kpc^-2]")
plt.legend(prop={"size": 10})
plt.show()

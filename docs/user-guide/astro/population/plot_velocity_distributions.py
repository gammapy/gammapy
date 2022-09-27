"""Plot velocity distributions of Galactic sources."""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.astro.population import velocity_distributions
from gammapy.utils.random import normalize

velocity = np.linspace(10, 3000, 200) * u.km / u.s

ax = plt.subplot()

for key in velocity_distributions:
    model = velocity_distributions[key]()
    label = model.__class__.__name__
    x = velocity.value
    y = normalize(model, velocity[0].value, velocity[-1].value)(velocity.value)
    ax.plot(x, y, linestyle="-", label=label)

ax.set_xlim(velocity[0].value, velocity[-1].value)
ax.set_ylim(0, 0.005)
ax.set_xlabel("Velocity [km/s]")
ax.set_ylabel("Probability Density [(km / s)^-1]")
ax.semilogx()
plt.legend(prop={"size": 10})
plt.show()

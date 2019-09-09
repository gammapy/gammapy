"""Plot velocity distributions of Galactic sources."""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.astro.population import velocity_distributions
from gammapy.utils.random import normalize

velocity = np.linspace(10, 3000, 200) * u.km / u.s

for key in velocity_distributions:
    model = velocity_distributions[key]()
    label = model.__class__.__name__
    x = velocity.value
    y = normalize(model, velocity[0].value, velocity[-1].value)(velocity.value)
    plt.plot(x, y, linestyle="-", label=label)

plt.xlim(velocity[0].value, velocity[-1].value)
plt.ylim(0, 0.005)
plt.xlabel("Velocity [km/s]")
plt.ylabel("Probability Density [(km / s)^-1]")
plt.semilogx()
plt.legend(prop={"size": 10})
plt.show()

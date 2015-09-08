"""Plot velocity distributions of Galactic sources."""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.astro.population import velocity_distributions
from gammapy.utils.distributions import normalize

# TODO: use Quantity
v_min, v_max = 10, 3000  # km / s
v = np.linspace(v_min, v_max, 200)

for key in velocity_distributions:
    model = velocity_distributions[key]()
    label = model.__class__.__name__
    plt.plot(v, normalize(model, v_min, v_max)(v), linestyle='-', label=label)

plt.xlim(v_min, v_max)
plt.ylim(0, 0.004)
plt.xlabel('Velocity [km/s]')
plt.ylabel('Probability Density [(km / s)^-1]')
plt.semilogx()
plt.legend(prop={'size': 10})
plt.show()

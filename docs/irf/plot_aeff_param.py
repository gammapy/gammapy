import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.irf import EffectiveAreaTable

energy = np.logspace(-3, 3, 100) * u.TeV

for instrument in ["HESS", "HESS2", "CTA"]:
    aeff = EffectiveAreaTable.from_parametrization(energy, instrument)
    ax = aeff.plot(label=instrument)

ax.set_yscale("log")
ax.set_xlim([1e-3, 1e3])
ax.set_ylim([1e3, 1e12])
plt.legend(loc="best")
plt.show()

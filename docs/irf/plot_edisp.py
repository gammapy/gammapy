"""Plot energy dispersion example."""
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from gammapy.irf import EnergyDispersion

ebounds = np.logspace(-1, 2, 101) * u.TeV
energy_dispersion = EnergyDispersion.from_gauss(
    e_true=ebounds, e_reco=ebounds, sigma=0.3,
)

energy_dispersion.plot_matrix()
plt.show()

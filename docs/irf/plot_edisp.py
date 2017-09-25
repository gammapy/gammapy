"""Plot energy dispersion example."""
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from gammapy.irf import EnergyDispersion

ebounds = np.logspace(-1, 2, 101) * u.TeV

edisp = EnergyDispersion.from_gauss(
    e_true=ebounds, e_reco=ebounds,
    bias=0, sigma=0.3,
)

edisp.peek()
plt.show()

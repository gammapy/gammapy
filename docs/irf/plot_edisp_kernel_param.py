"""Plot an energy dispersion using a gaussian parametrisation"""
import numpy as np
import astropy.units as u
from gammapy.irf import EDispKernel
import matplotlib.pyplot as plt

energy = np.logspace(0, 1, 101) * u.TeV
edisp = EDispKernel.from_gauss(
    energy=energy, energy_true=energy,
    sigma=0.1, bias=0,
)
edisp.peek()
plt.show()
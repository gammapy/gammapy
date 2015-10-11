"""Plot energy dispersion example."""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.irf import EnergyDispersion
from gammapy.spectrum import EnergyBounds

ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
energy_dispersion = EnergyDispersion.from_gauss(ebounds, sigma=0.3)

energy_dispersion.plot(type='matrix')
plt.show()

"""Plot energy dispersion example."""
import numpy as np
import matplotlib.pyplot as plt
from gammapy import irf
from gammapy.spectrum import EnergyBounds

ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
pdf_matrix = irf.gauss_energy_dispersion_matrix(ebounds, sigma=0.2)
energy_dispersion = irf.EnergyDispersion(pdf_matrix, ebounds)

energy_dispersion.plot(type='matrix')
plt.show()

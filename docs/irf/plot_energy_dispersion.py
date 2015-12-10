"""Plot energy dispersion example."""
import matplotlib.pyplot as plt
from gammapy.irf import EnergyDispersion
from gammapy.utils.energy import EnergyBounds

ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
energy_dispersion = EnergyDispersion.from_gauss(ebounds, sigma=0.3)

energy_dispersion.plot_matrix()
plt.show()

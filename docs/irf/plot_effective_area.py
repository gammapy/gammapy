# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Plot approximate effective area for HESS, HESS2 and CTA.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Quantity
from gammapy.irf import abramowski_effective_area

energy = Quantity(np.logspace(-3, 3, 100), 'TeV')

for instrument in ['HESS', 'HESS2', 'CTA']:
    a_eff = abramowski_effective_area(energy, instrument)
    plt.plot(energy.value, a_eff.value, label=instrument)

plt.loglog()
plt.xlabel('Energy (TeV)')
plt.ylabel('Effective Area (cm^2)')
plt.xlim([1e-3, 1e3])
plt.ylim([1e3, 1e12])
plt.legend(loc='best')
plt.show()

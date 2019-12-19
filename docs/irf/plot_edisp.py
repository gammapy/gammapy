"""Plot energy dispersion example."""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.irf import EDispKernel

ebounds = np.logspace(-1, 2, 101) * u.TeV

edisp = EDispKernel.from_gauss(e_true=ebounds, e_reco=ebounds, bias=0, sigma=0.3)

edisp.peek()
plt.show()

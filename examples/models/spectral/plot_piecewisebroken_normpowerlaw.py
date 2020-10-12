r"""
.. _piecewise-broken-powerlaw-norm-spectral:

Piecewise Broken Power Law Norm Spectral Model
==============================================

This model parametrises a piecewise broken power law 
with a free norm parameter at each fixed energy node.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import  PiecewiseBrokenPowerLawNormSpectralModel

energy_range = [0.1, 100] * u.TeV
model = PiecewiseBrokenPowerLawNormSpectralModel(
    energy=[0.1, 1, 3, 10, 30, 100] * u.TeV, norms=[1, 3, 8, 10, 8, 2],
)
model.plot(energy_range, flux_unit="")
plt.grid(which="both")


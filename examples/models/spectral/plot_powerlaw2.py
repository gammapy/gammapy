r"""
.. _powerlaw2-spectral-model:

Power Law 2 Spectral Model
==========================

This model parametrises a power law spectrum with integral as amplitude parameter.

It is defined by the following equation:

.. math::
    \phi(E) = F_0 \cdot \frac{\Gamma + 1}{E_{0, max}^{-\Gamma + 1}
     - E_{0, min}^{-\Gamma + 1}} \cdot E^{-\Gamma}

See also: https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

import matplotlib.pyplot as plt
from astropy import units as u
from gammapy.modeling.models import (
    Models,
    SkyModel,
    PowerLaw2SpectralModel,
)

energy_range = [0.1, 100] * u.TeV
model = PowerLaw2SpectralModel(
    amplitude=u.Quantity(2.9227116204223784, "cm-2 s-1"),
    index=2.3 * u.Unit(""),
    emin=1 * u.TeV,
    emax=10 * u.TeV,
)
model.plot(energy_range)
plt.grid(which="both");

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model)
models = Models([model])

print(models.to_yaml())

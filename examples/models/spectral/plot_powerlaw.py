r"""
.. _powerlaw-spectral-model:

Powerlaw Spectral Model
=======================

This model parametrises a power law spectrum.

It is defined by the following equation:

.. math::
    \phi(E) = \phi_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

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
    PowerLawSpectralModel,
)

energy_range = [0.1, 100] * u.TeV
model = PowerLawSpectralModel(
    index=1 * u.Unit(""),
    amplitude=2 / u.cm ** 2 / u.s / u.TeV,
    reference=1 * u.TeV,
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

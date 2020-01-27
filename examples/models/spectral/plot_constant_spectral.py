r"""
.. _constant-spectral-model:

Constant Spectral Model
=======================

This model takes a constant value along the spectral range.

    .. math:: \phi(E) = k
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
    ConstantSpectralModel,
)

energy_range = [0.1, 100] * u.TeV
k = ConstantSpectralModel(const="1 / (cm2 s TeV)")
k.plot(energy_range)
plt.grid(which="both");

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=k)
models = Models([model])

print(models.to_yaml())

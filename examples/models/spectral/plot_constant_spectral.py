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

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import ConstantSpectralModel, Models, SkyModel

energy_range = [0.1, 100] * u.TeV
model = ConstantSpectralModel(const="1 / (cm2 s TeV)")
model.plot(energy_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="constant-model")
models = Models([model])

print(models.to_yaml())

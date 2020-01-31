r"""
.. _gaussian-spectral-model:

Gaussian Spectral Model
=======================

This model parametrises a gaussian spectrum.

It is defined by the following equation:

.. math::
    \phi(E) = \frac{N_0}{\sigma \sqrt{2\pi}}  \exp{ \frac{- \left( E-\bar{E} \right)^2 }{2 \sigma^2} }

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import GaussianSpectralModel, Models, SkyModel

energy_range = [0.1, 100] * u.TeV
model = GaussianSpectralModel(
    norm=4 / u.cm ** 2 / u.s, mean=2 * u.TeV, sigma=0.2 * u.TeV
)
model.plot(energy_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="gaussian-model")
models = Models([model])

print(models.to_yaml())

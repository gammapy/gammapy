r"""
.. _gaussian-temporal-model:

Gaussian temporal model
=======================

This model parametrises a gaussian time model.


.. math::
    F(t) = exp(-0.5* \frac{ (t - t_{ref})^2 } { \sigma^2 })
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    GaussianTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

sigma = "3 h"
t_ref = Time("2020-10-01")
time_range = [t_ref - 0.5 * u.d, t_ref + 0.5 * u.d]
gaussian_model = GaussianTemporalModel(t_ref=t_ref.mjd * u.d, sigma=sigma)
gaussian_model.plot(time_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=gaussian_model,
    name="gaissian_model",
)
models = Models([model])

print(models.to_yaml())

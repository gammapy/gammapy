r"""
.. _powerlaw-temporal-model:

PowerLaw temporal model
=======================

This model parametrises a power-law time model.

.. math:: F(t) = ((t - t_{ref})/t_0)^alpha

"""


# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    Models,
    PowerLawSpectralModel,
    PowerLawTemporalModel,
    SkyModel,
)

time_range = [Time.now(), Time.now() + 2 * u.d]
pl_model = PowerLawTemporalModel(alpha=-2.0, t_ref=(time_range[0].mjd - 0.1) * u.d)
pl_model.plot(time_range)
plt.grid(which="both")
plt.yscale("log")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=pl_model,
    name="powerlaw-model",
)
models = Models([model])

print(models.to_yaml())

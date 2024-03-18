r"""
.. _constant-temporal-model:

Constant temporal model
=======================

This model parametrises a constant time model.

.. math:: F(t) = k

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    ConstantTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

time_range = [Time.now(), Time.now() + 1 * u.d]
constant_model = ConstantTemporalModel(const=1)
constant_model.plot(time_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=constant_model,
    name="constant-model",
)
models = Models([model])

print(models.to_yaml())

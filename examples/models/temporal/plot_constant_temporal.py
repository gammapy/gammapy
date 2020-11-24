r"""
.. _constant-temporal-model:

Constant Temporal Model
=======================

This model parametrises a constant time model.

.. math:: F(E) = k

"""


# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:
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


model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=constant_model,
    name="constant-model",
)
models = Models([model])

print(models.to_yaml())

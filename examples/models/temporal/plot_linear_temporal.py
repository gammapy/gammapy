r"""
.. _linear-temporal-model:

Linear temporal model
=======================

This model parametrises a linear time model.

.. math:: F(t) = \alpha + \beta \cdot (t - t_{\rm{ref}})

"""


# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    LinearTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

time_range = [Time.now(), Time.now() + 2 * u.d]
linear_model = LinearTemporalModel(
    alpha=1, beta=0.5 / u.d, t_ref=(time_range[0].mjd - 0.1) * u.d
)
linear_model.plot(time_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=linear_model,
    name="linear-model",
)
models = Models([model])

print(models.to_yaml())

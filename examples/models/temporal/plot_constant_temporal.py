r"""
.. _constant-temporal-model:

Constant Temporal Model
=======================

This model parametrises a constant time model.


"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import Models, ConstantTemporalModel, SkyModel

time_range = [Time.now(), Time.now() + 1 * u.d]
model = ConstantTemporalModel(const=1)
model.plot(time_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

#model = SkyModel(spectral_model=model, name="power-law-model")
#models = Models([model])

#print(models.to_yaml())

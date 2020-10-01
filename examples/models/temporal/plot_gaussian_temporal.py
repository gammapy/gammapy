r"""
.. _gaussian-temporal-model:

Gaussian Temporal Model
=======================

This model parametrises a gaussian time model.


"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import Models, GaussianTemporalModel, SkyModel
sigma = "3 h"
t_ref = Time("2020-10-01")
time_range = [t_ref - 0.5 * u.d , t_ref + 0.5 * u.d]
model = GaussianTemporalModel(t_ref = t_ref.mjd * u.d, sigma=sigma)
model.plot(time_range)
plt.grid(which="both")






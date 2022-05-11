r"""
.. _generalized-gaussian-temporal-model:

Generalized Gaussian temporal model
===================================

This model parametrises a generalized Gaussian time model.


.. math::
        F(t) = exp( - 0.5 * (\frac{|t - t_{\rm{ref}}|}{t_{\rm{rise}}}) ^ {1 / \eta}) \text{ for } t < t_{\rm{ref}}
            
        F(t) = exp( - 0.5 * (\frac{|t - t_{\rm{ref}}|}{t_{\rm{decay}}}) ^ {1 / \eta}) \text{ for } t > t_{\rm{ref}}    
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    GeneralizedGaussianTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

t_rise = Quantity(0.1, "d")
t_decay = Quantity(1, "d")
eta = Quantity(2 / 3, "")
t_ref = Time("2020-10-01")
time_range = [t_ref - 1 * u.d, t_ref + 1 * u.d]
gen_gaussian_model = GeneralizedGaussianTemporalModel(
    t_ref=t_ref.mjd * u.d, t_rise=t_rise, t_decay=t_decay, eta=eta
)
gen_gaussian_model.plot(time_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=gen_gaussian_model,
    name="generalized_gaussian_model",
)
models = Models([model])

print(models.to_yaml())

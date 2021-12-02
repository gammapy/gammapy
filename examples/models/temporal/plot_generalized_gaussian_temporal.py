# -*- coding: utf-8 -*-
r"""
.. _generalized-gaussian-temporal-model:
Generalized Gaussian temporal model
=======================
This model parametrises a generalized Gaussian time model.
.. math::
        F(t) = exp( - (\frac{|t - t_{ref}|}{\sigma_rise}) ^ nu)   for  t < t_ref
        
        F(t) = exp( - (\frac{|t - t_{ref}|}{\sigma_decay}) ^ nu)   for  t > t_ref
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:


from astropy import units as u
from astropy.units import Quantity
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import (GeneralizedGaussianTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel)

sigma_rise = Quantity(0.1, "h")
sigma_decay = Quantity(1, "h")
nu = Quantity(1.5, "")
t_ref = Time("2020-10-01")
time_range = [t_ref - 1 * u.d, t_ref + 1 * u.d]
gen_gaussian_model = GeneralizedGaussianTemporalModel(t_ref = t_ref.mjd * u.d,\
                                                  sigma_rise = sigma_rise,\
                                                  sigma_decay = sigma_decay,\
                                                  nu = nu)
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

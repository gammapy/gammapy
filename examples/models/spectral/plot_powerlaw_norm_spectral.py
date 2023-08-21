r"""
.. _powerlaw-spectral-norm-model:

Power law norm spectral model
========================

This model parametrises a power law spectral correction with a norm and tilt parameter.

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    Models,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
)

energy_bounds = [0.1, 100] * u.TeV

model = PowerLawSpectralModel()
norm = PowerLawNormSpectralModel(
    norm=5,
    reference=1 * u.TeV,
)
pwl_norm = model * norm
model.plot(energy_bounds, label="PowerLaw")
pwl_norm.plot(energy_bounds, label="PowerLaw with a norm correction")
plt.legend(loc="best")
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=pwl_norm, name="power-law-norm-model")
models = Models([model])

print(models.to_yaml())

r"""
.. _logparabola-spectral-norm-model:

Log parabola spectral norm model
===========================

This model parametrises a log parabola spectral correction with a norm parameter.

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    LogParabolaNormSpectralModel,
    LogParabolaSpectralModel,
    Models,
    SkyModel,
)

energy_bounds = [0.1, 100] * u.TeV

model = LogParabolaSpectralModel()
norm = LogParabolaNormSpectralModel(
    norm=1.5,
    reference=1 * u.TeV,
)
lp_norm = model * norm
model.plot(energy_bounds, label="LogParabola")
lp_norm.plot(energy_bounds, label="LogParabola with a norm correction")
plt.legend(loc="best")
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=lp_norm, name="log-parabola-norm-model")
models = Models([model])

print(models.to_yaml())

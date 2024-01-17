r"""
.. _logparabola-spectral-norm-model:

Log parabola spectral norm model
================================

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
    TemplateSpectralModel,
)

energy_bounds = [0.1, 100] * u.TeV

energy = [0.3, 1, 3, 10, 30] * u.TeV
values = [40, 30, 20, 10, 1] * u.Unit("TeV-1 s-1 cm-2")
template = TemplateSpectralModel(energy, values)
norm = LogParabolaNormSpectralModel(
    norm=1.5,
    reference=1 * u.TeV,
)

template.plot(energy_bounds=energy_bounds, label="Template model")
lp_norm = template * norm
lp_norm.plot(energy_bounds, label="Template model with LogParabola norm correction")
plt.legend(loc="best")
plt.grid(which="both")


# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=lp_norm, name="log-parabola-norm-model")
models = Models([model])

print(models.to_yaml())

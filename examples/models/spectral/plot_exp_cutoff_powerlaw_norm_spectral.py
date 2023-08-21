r"""
.. _exp-cutoff-powerlaw-norm-spectral-model:

Exponential cutoff power law norm spectral model
========================

This model parametrises a cutoff power law spectral correction with a norm parameter.

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    ExpCutoffPowerLawNormSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    Models,
    SkyModel,
)

energy_bounds = [0.1, 100] * u.TeV

model = ExpCutoffPowerLawSpectralModel()
norm = ExpCutoffPowerLawNormSpectralModel(
    norm=2,
    reference=1 * u.TeV,
)
ecpl_norm = model * norm
model.plot(energy_bounds, label="Cutoff PowerLaw")
ecpl_norm.plot(energy_bounds, label="Cutoff PowerLaw with a norm correction")
plt.legend(loc="best")
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=ecpl_norm, name="exp-cutoff-power-law-norm-model")
models = Models([model])

print(models.to_yaml())

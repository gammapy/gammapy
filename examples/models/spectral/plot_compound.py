r"""
.. _compound-spectral-model:

Compound spectral model
=======================

This model is formed by the arithmetic combination of any two other spectral models.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

import operator
from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    CompoundSpectralModel,
    LogParabolaSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

energy_bounds = [0.1, 100] * u.TeV
pwl = PowerLawSpectralModel(
    index=2.0, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
lp = LogParabolaSpectralModel(
    amplitude="1e-12 cm-2 s-1 TeV-1", reference="10 TeV", alpha=2.0, beta=1.0
)
model = CompoundSpectralModel(pwl, lp, operator.add)
model.plot(energy_bounds)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="compound-model")
models = Models([model])

print(models.to_yaml())

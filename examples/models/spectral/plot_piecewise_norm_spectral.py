r"""
.. _piecewise-norm-spectral:

Piecewise  norm spectral model
==============================

This model parametrises a piecewise spectral correction
with a free norm parameter at each fixed energy node.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    Models,
    PiecewiseNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
)

energy_bounds = [0.1, 100] * u.TeV
model = PiecewiseNormSpectralModel(
    energy=[0.1, 1, 3, 10, 30, 100] * u.TeV,
    norms=[1, 3, 8, 10, 8, 2],
)
model.plot(energy_bounds, yunits=u.Unit(""))
plt.grid(which="both")


# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = model * PowerLawSpectralModel()
model = SkyModel(spectral_model=model, name="piecewise-norm-model")
models = Models([model])

print(models.to_yaml())

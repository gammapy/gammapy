r"""
.. _exp-cutoff-powerlaw-3fgl-spectral-model:

Exponential Cutoff Powerlaw Spectral Model used for 3FGL
========================================================

This model is defined by the following equation:

.. math::
    \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma}
              \exp \left( \frac{E_0 - E}{E_{C}} \right)

Note that the parametrization is different from `ExpCutoffPowerLawSpectralModel`
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

import matplotlib.pyplot as plt
from astropy import units as u
from gammapy.modeling.models import (
    Models,
    SkyModel,
    ExpCutoffPowerLaw3FGLSpectralModel,
)

energy_range = [0.1, 100] * u.TeV
model = ExpCutoffPowerLaw3FGLSpectralModel(
    index=2.3 * u.Unit(""),
    amplitude=4 / u.cm ** 2 / u.s / u.TeV,
    reference=1 * u.TeV,
    ecut=10 * u.TeV,
)
model.plot(energy_range)
plt.grid(which="both");

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model)
models = Models([model])

print(models.to_yaml())

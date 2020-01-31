r"""
.. _super-exp-cutoff-powerlaw-3fgl-spectral-model:

Super Exponential Cutoff Power Law Model used for 3FGL
======================================================

This model parametrises super exponential cutoff power-law model spectrum used for 3FGL.

It is defined by the following equation:

.. math::
    \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma_1}
              \exp \left( \left(\frac{E_0}{E_{C}} \right)^{\Gamma_2} -
                          \left(\frac{E}{E_{C}} \right)^{\Gamma_2}
                          \right)
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    Models,
    SkyModel,
    SuperExpCutoffPowerLaw3FGLSpectralModel,
)

energy_range = [0.1, 100] * u.TeV
model = SuperExpCutoffPowerLaw3FGLSpectralModel(
    index_1=1,
    index_2=2,
    amplitude="1e-12 TeV-1 s-1 cm-2",
    reference="1 TeV",
    ecut="10 TeV",
)
model.plot(energy_range)
plt.grid(which="both")
plt.ylim(1e-24, 1e-10)

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="super-exp-cutoff-power-law-3fgl-model")
models = Models([model])

print(models.to_yaml())

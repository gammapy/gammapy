r"""
.. _super-exp-cutoff-powerlaw-4fgl-dr3-spectral-model:

Super Exponential Cutoff Power Law Model used for 4FGL-DR3
==========================================================

This model parametrises super exponential cutoff power-law model spectrum used for 4FGL.

It is defined by the following equation:

.. math::


    \phi(e) =
            \begin{cases}
                \phi_0 \cdot \left(\frac{E}{E_0}\right)^{\frac{\a}{\Gamma_2} -\Gamma_1} \cdot \exp \left(
                  \frac{\a}{\Gamma_2^2} \left( 1 - \left(\frac{E}{E_0}\right)^{\frac{\a}{\Gamma_2} \right)
              \right)&
                              \\
                \phi_0 \cdot \left(\frac{E}{E_0}\right)^{ -\Gamma_1 - \frac{\a}{2} \ln \frac{E}{E_0} - \frac{\a \Gamma_2}{6} \ln^2 \frac{E}{E_0} - \frac{\a \Gamma_2^2}{24} \ln^3 \frac{E}{E_0}}\\
                0 & \text{for } \left| \Gamma_2 \ln \frac{E}{E_0}  \right|
            \end{cases}

See Equation (2) and (3) in https://arxiv.org/pdf/2201.11184.pdf
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
    SuperExpCutoffPowerLaw4FGLDR3SpectralModel,
)

energy_range = [0.1, 100] * u.TeV
model = SuperExpCutoffPowerLaw4FGLDR3SpectralModel(
    index_1=1,
    index_2=2,
    amplitude="1e-12 TeV-1 cm-2 s-1",
    reference="1 TeV",
    expfactor=1e-2,
)
model.plot(energy_range)
plt.grid(which="both")
plt.ylim(1e-24, 1e-10)

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="super-exp-cutoff-power-law-4fgl-dr3-model")
models = Models([model])

print(models.to_yaml())

r"""
.. _broken-powerlaw-spectral-model:

Broken power law spectral model
===============================

This model parametrises a broken power law spectrum.

It is defined by the following equation:

.. math::
    \phi(E) = \phi_0 \cdot \begin{cases}
                          \left( \frac{E}{E_{break}} \right)^{-\Gamma1} & \text{if } E < E_{break} \\
                          \left( \frac{E}{E_{break}} \right)^{-\Gamma2} & \text{otherwise}
                         \end{cases}
    """

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import BrokenPowerLawSpectralModel, Models, SkyModel

energy_bounds = [0.1, 100] * u.TeV
model = BrokenPowerLawSpectralModel(
    index1=1.5,
    index2=2.5,
    amplitude="1e-12 TeV-1 cm-2 s-1",
    ebreak="1 TeV",
)
model.plot(energy_bounds)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="broken-power-law-model")
models = Models([model])

print(models.to_yaml())

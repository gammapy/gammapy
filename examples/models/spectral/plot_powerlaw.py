r"""
.. _powerlaw-spectral-model:

Power Law Spectral Model
========================

This model parametrises a power law spectrum.

It is defined by the following equation:

.. math::
    \phi(E) = \phi_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import Models, PowerLawSpectralModel, SkyModel

energy_range = [0.1, 100] * u.TeV
model = PowerLawSpectralModel(
    index=2, amplitude="1e-12 TeV-1 cm-2 s-1", reference=1 * u.TeV,
)
model.plot(energy_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="power-law-model")
models = Models([model])

print(models.to_yaml())

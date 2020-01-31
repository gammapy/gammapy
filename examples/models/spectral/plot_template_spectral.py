r"""
.. _template-spectral-model:

Template Spectral Model
=======================

This model is defined by custom tabular values.

The units returned will be the units of the values array provided at
initialization. The model will return values interpolated in
log-space, returning 0 for energies outside of the limits of the provided
energy array.

The class implementation follows closely what has been done in
`naima.models.TemplateSpectralModel`
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import Models, SkyModel, TemplateSpectralModel

energy_range = [0.1, 1] * u.TeV
energy = np.array([1e6, 3e6, 1e7, 3e7]) * u.MeV
values = np.array([4.4e-38, 2.0e-38, 8.8e-39, 3.9e-39]) * u.Unit("MeV-1 s-1 cm-2")
model = TemplateSpectralModel(energy=energy, values=values)
model.plot(energy_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="template-model")
models = Models([model])

print(models.to_yaml())

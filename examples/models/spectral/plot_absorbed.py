r"""
.. _absorbed-spectral-model:

Absorbed Spectral Model
=======================

This model evaluates absorbed spectral model.

The spectral model is evaluated, and then multiplied with an EBL
absorption factor given by

.. math::
    \exp{ \left ( -\alpha \times \tau(E, z) \right )}

where :math:`\tau(E, z)` is the optical depth predicted by the model
(`~gammapy.modeling.models.Absorption`), which depends on the energy of the gamma-rays and the
redshift z of the source, and :math:`\alpha` is a scale factor
(default: 1) for the optical depth.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    AbsorbedSpectralModel,
    Absorption,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

redshift = 0.117
absorption = Absorption.read_builtin("dominguez")

# Spectral model corresponding to PKS 2155-304 (quiescent state)
index = 3.53
amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
reference = 1 * u.TeV
pwl = PowerLawSpectralModel(index=index, amplitude=amplitude, reference=reference)

# EBL + PWL model
model = AbsorbedSpectralModel(
    spectral_model=pwl, absorption=absorption, redshift=redshift
)

energy_range = [0.1, 100] * u.TeV
model.plot(energy_range)
plt.grid(which="both")
plt.ylim(1e-24, 1e-8)

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="absorbed-model")
models = Models([model])

print(models.to_yaml())

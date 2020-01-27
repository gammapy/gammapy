r"""
.. _absorbed-spectral-model:

Absorbed Spectral Model
=======================

The spectral model is evaluated, and then multiplied with an EBL
absorption factor given by

.. math::
    \exp{ \left ( -\alpha \times \tau(E, z) \right )}

where :math:`\tau(E, z)` is the optical depth predicted by the model
(`Absorption`), which depends on the energy of the gamma-rays and the
redshift z of the source, and :math:`\alpha` is a scale factor
(default: 1) for the optical depth.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from gammapy.modeling.models import (
    Models,
    SkyModel,
    Absorption,
    AbsorbedSpectralModel,
    PowerLawSpectralModel
)

redshift = 0.117
absorption = Absorption.read_builtin("dominguez")

# Spectral model corresponding to PKS 2155-304 (quiescent state)
index = 3.53
amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
reference = 1 * u.TeV
pwl = PowerLawSpectralModel(index=index, amplitude=amplitude, reference=reference)

# EBL + PWL model
absorbed = AbsorbedSpectralModel(
    spectral_model=pwl, absorption=absorption, parameter=redshift
)

energy_range = [0.1, 100] * u.TeV
absorbed.plot(energy_range);

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=absorbed)
models = Models([model])

print(models.to_yaml())

r"""
.. _absorption-spectral-model:

EBL absorption spectral model
=============================

This model evaluates absorbed spectral model.

The EBL absorption factor given by

.. math::
    \exp{ \left ( -\alpha \times \tau(E, z) \right )}

where :math:`\tau(E, z)` is the optical depth predicted by the model
(`~gammapy.modeling.models.EBLAbsorptionNormSpectralModel`), which depends on the energy of the gamma-rays and the
redshift z of the source, and :math:`\alpha` is a scale factor
(default: 1) for the optical depth.

The available EBL models are defined in `~gammapy.modeling.models.EBL_DATA_BUILTIN`.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    EBL_DATA_BUILTIN,
    EBLAbsorptionNormSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

# Print the available EBL models
print(EBL_DATA_BUILTIN.keys())

# Here we illustrate how to create and plot EBL absorption models for a redshift of 0.5
# sphinx_gallery_thumbnail_number = 1

redshift = 0.5
dominguez = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=redshift)
franceschini = EBLAbsorptionNormSpectralModel.read_builtin(
    "franceschini", redshift=redshift
)
finke = EBLAbsorptionNormSpectralModel.read_builtin("finke", redshift=redshift)
franceschini17 = EBLAbsorptionNormSpectralModel.read_builtin(
    "franceschini17", redshift=redshift
)
saldana21 = EBLAbsorptionNormSpectralModel.read_builtin(
    "saldana-lopez21", redshift=redshift
)

fig, (ax_ebl, ax_model) = plt.subplots(
    nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={"left": 0.08, "right": 0.96}
)

energy_bounds = [0.08, 3] * u.TeV
opts = dict(energy_bounds=energy_bounds, xunits=u.TeV)

franceschini.plot(ax=ax_ebl, label="Franceschini 2008", **opts)
finke.plot(ax=ax_ebl, label="Finke 2010", **opts)
dominguez.plot(ax=ax_ebl, label="Dominguez 2011", **opts)
franceschini17.plot(ax=ax_ebl, label="Franceschni 2017", **opts)
saldana21.plot(ax=ax_ebl, label="Saldana-Lopez 2021", **opts)

ax_ebl.set_ylabel(r"Absorption coefficient [$\exp{(-\tau(E))}$]")
ax_ebl.set_xlim(energy_bounds.value)
ax_ebl.set_ylim(1e-4, 2)
ax_ebl.set_title(f"EBL models (z={redshift})")
ax_ebl.grid(which="both")
ax_ebl.legend(loc="best")


# Spectral model corresponding to PKS 2155-304 (quiescent state)
index = 3.53
amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
reference = 1 * u.TeV
pwl = PowerLawSpectralModel(index=index, amplitude=amplitude, reference=reference)

# The power-law model is multiplied by the EBL norm spectral model
redshift = 0.117
absorption = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=redshift)

model = pwl * absorption

energy_bounds = [0.1, 100] * u.TeV

model.plot(energy_bounds, ax=ax_model)
ax_model.grid(which="both")
ax_model.set_ylim(1e-24, 1e-8)
ax_model.set_title("Absorbed Power Law")
plt.show()

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="absorbed-model")
models = Models([model])

print(models.to_yaml())

r"""
.. _PhaseCurve-temporal-model:

Phase curve temporal model
==========================

This model parametrises a PhaseCurve time model, i.e. with a template phasogram and timing parameters

"""

import astropy.units as u
from astropy.time import Time
from gammapy.modeling.models import (
    Models,
    PowerLawSpectralModel,
    SkyModel,
    TemplatePhaseCurveTemporalModel,
)

path = "$GAMMAPY_DATA/tests/phasecurve_LSI_DC.fits"
t_ref = 43366.275 * u.d
f0 = 1.0 / (26.7 * u.d)

phase_model = TemplatePhaseCurveTemporalModel.read(path, t_ref, 0.0, f0)
time_range = [Time("59100", format="mjd"), Time("59200", format="mjd")]

phase_model.plot(time_range, n_points=400)

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=phase_model,
    name="phase_curve_model",
)
models = Models([model])

print(models.to_yaml())

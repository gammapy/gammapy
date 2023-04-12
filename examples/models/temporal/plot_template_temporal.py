r"""
.. _LightCurve-temporal-model:

Light curve temporal model
==========================

This model parametrises a LightCurve time model.

The gammapy internal lightcurve model format is a `~gammapy.maps.RegionNDMap`
with `time`, and optionally `energy` axes. The times are defined wrt to a reference time.

For serialisation, a `table` and a `map` format are supported.
A `table` format is a `~astropy.table.Table` with the reference_time`
serialised as a dictionary in the table meta. Only maps without an energy axis can
be serialised to this format.

In `map` format, a `~gammapy.maps.RegionNDMap` is serialised, with the `reference_time`
in the SKYMAP_BANDS HDU.
"""


from astropy.time import Time
from gammapy.modeling.models import (
    LightCurveTemplateTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

time_range = [Time("59100", format="mjd"), Time("59365", format="mjd")]
path = "$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits"
light_curve_model = LightCurveTemplateTemporalModel.read(path)
light_curve_model.plot(time_range)

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(
    spectral_model=PowerLawSpectralModel(),
    temporal_model=light_curve_model,
    name="light_curve_model",
)
models = Models([model])

print(models.to_yaml())

r"""
.. _point-spatial-model:

Point Spatial Model
===================

This model is a delta function centered in *lon_0* and *lat_0* parameters provided:

.. math:: \phi(lon, lat) = \delta{(lon - lon_0, lat - lat_0)}

The model is defined on the celestial sphere in the coordinate frame provided by the user.
"""

#%%
# Example plot
# ------------
# Here is an example plot of the model:

from gammapy.modeling.models import (
    PointSpatialModel,
    SkyModel,
    Models,
    PowerLawSpectralModel,
)

model = PointSpatialModel(
    lon_0="23 deg", lat_0="32 deg", frame="galactic",
)

ax = model.plot()

#%%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

pwl = PowerLawSpectralModel()
point = PointSpatialModel()

model = SkyModel(spectral_model=pwl, spatial_model=point)
models = Models([model])

print(models.to_yaml())

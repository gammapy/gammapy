r"""
.. _constant-spatial-model:

Constant Spatial Model
======================

This model is a spatially constant model.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from gammapy.maps import WcsGeom
from gammapy.modeling.models import (
    Models,
    ConstantSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)

geom = WcsGeom.create(npix=(100,100), binsz=0.1)
model = ConstantSpatialModel(value="42 sr-1")
model.plot(geom=geom)

#%%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

pwl = PowerLawSpectralModel()
constant = ConstantSpatialModel()

model = SkyModel(spectral_model=pwl, spatial_model=constant)
models = Models([model])

print(models.to_yaml())

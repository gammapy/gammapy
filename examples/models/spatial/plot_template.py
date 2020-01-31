r"""
.. _template-spatial-model:

Template Spatial Model
======================

This is a spatial model based on a 2D sky map provided as a template.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from gammapy.maps import Map
from gammapy.modeling.models import (
    Models,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)

filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"

m = Map.read(filename)
model = TemplateSpatialModel(m)

model.plot(add_cbar=True)

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

pwl = PowerLawSpectralModel()
template = TemplateSpatialModel(m)

model = SkyModel(spectral_model=pwl, spatial_model=template, name="pwl-template-model")
models = Models([model])

print(models.to_yaml())

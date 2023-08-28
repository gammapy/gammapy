r"""
.. _piecewise-norm-spatial:

Piecewise norm spatial model
============================

This model parametrises a piecewise spatial correction
with a free norm parameter at each fixed node in longitude, latitude
and optionaly energy.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

import numpy as np
from astropy import units as u
from gammapy.maps import MapCoord, WcsGeom
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    PiecewiseNormSpatialModel,
)

geom = WcsGeom.create(skydir=(50, 0), npix=(120, 120), binsz=0.03, frame="galactic")
coords = MapCoord.create(geom.footprint)
coords["lon"] *= u.deg
coords["lat"] *= u.deg

model = PiecewiseNormSpatialModel(
    coords, norms=np.array([0.5, 3, 2, 1]), frame="galactic"
)

model.plot(geom=geom)


# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

bkg_model = FoVBackgroundModel(spatial_model=model, dataset_name="dataset")
models = Models([bkg_model])

print(models.to_yaml())

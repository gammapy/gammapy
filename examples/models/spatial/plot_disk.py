r"""
.. _disk-spatial-model:

Disk Spatial Model
==================

This is a spatial model parametrising a disk.

By default, the model is symmetric, i.e. a disk:

.. math::

    \phi(lon, lat) = \frac{1}{2 \pi (1 - \cos{r_0}) } \cdot
            \begin{cases}
                1 & \text{for } \theta \leq r_0 \
                0 & \text{for } \theta > r_0
            \end{cases}

where :math:`\theta` is the sky separation. To improve fit convergence of the
model, the sharp edges is smoothed using `~scipy.special.erf`.

In case an eccentricity (`e`) and rotation angle (:math:`\phi`) are passed,
then the model is an elongated disk (i.e. an ellipse), with a major semiaxis of length :math:`r_0`
and position angle :math:`\phi` (increaing counter-clockwise from the North direction).

The model is defined on the celestial sphere, with a normalization defined by:

.. math::

    \int_{4\pi}\phi(\text{lon}, \text{lat}) \,d\Omega = 1\,.
"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:


import numpy as np
from astropy.coordinates import Angle
from gammapy.modeling.models import (
    DiskSpatialModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

phi = Angle("30 deg")
model = DiskSpatialModel(
    lon_0="2 deg", lat_0="2 deg", r_0="1 deg", e=0.8, phi="30 deg", frame="galactic",
)

ax = model.plot(add_cbar=True)

# illustrate size parameter
region = model.to_region().to_pixel(ax.wcs)
artist = region.as_artist(facecolor="none", edgecolor="red")
ax.add_artist(artist)

transform = ax.get_transform("galactic")
ax.scatter(2, 2, transform=transform, s=20, edgecolor="red", facecolor="red")
ax.text(1.7, 1.85, r"$(l_0, b_0)$", transform=transform, ha="center")
ax.plot([2, 2 + np.sin(phi)], [2, 2 + np.cos(phi)], color="r", transform=transform)
ax.vlines(x=2, color="r", linestyle="--", transform=transform, ymin=0, ymax=5)
ax.text(2.15, 2.3, r"$\phi$", transform=transform)

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

pwl = PowerLawSpectralModel()
gauss = DiskSpatialModel()

model = SkyModel(spectral_model=pwl, spatial_model=gauss, name="pwl-disk-model")
models = Models([model])

print(models.to_yaml())

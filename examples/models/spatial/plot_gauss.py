"""
Gaussian Spatial Model
======================
Plot the gaussian spatial model.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import GaussianSpatialModel


m_geom = WcsGeom.create(
    binsz=0.01, width=(5, 5), skydir=(2, 2), coordsys="GAL", proj="AIT"
)
phi = Angle("30 deg")
model = GaussianSpatialModel(
    lon_0="2 deg",
    lat_0="2 deg",
    sigma="1 deg",
    e=0.7,
    phi=phi,
    frame="galactic",
)

coords = m_geom.get_coord()
vals = model(coords.lon, coords.lat)
skymap = Map.from_geom(m_geom, data=vals.value)

_, ax, _ = skymap.smooth("0.05 deg").plot()

transform = ax.get_transform("galactic")
ax.scatter(2, 2, transform=transform, s=20, edgecolor="red", facecolor="red")
ax.text(1.5, 1.85, r"$(l_0, b_0)$", transform=transform, ha="center")
ax.plot(
    [2, 2 + np.sin(phi)], [2, 2 + np.cos(phi)], color="r", transform=transform
)
ax.vlines(x=2, color="r", linestyle="--", transform=transform, ymin=-5, ymax=5)
ax.text(2.25, 2.45, r"$\phi$", transform=transform)
ax.contour(skymap.data, cmap="coolwarm", levels=10, alpha=0.6)
plt.show()
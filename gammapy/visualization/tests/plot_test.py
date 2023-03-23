import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from gammapy.maps import MapAxis, WcsNDMap
from gammapy.visualization.utils import plot_distribution

pos = SkyCoord(0, 0, frame="galactic", unit="deg")
axis = MapAxis.from_bounds(1, 3, 3, name="dummy")
nd_map = WcsNDMap.create(skydir=pos, width=4 * u.deg, binsz=0.04 * u.deg, axes=[axis])
nd_map_2 = WcsNDMap.create(skydir=pos, width=4 * u.deg, binsz=0.04 * u.deg, axes=[axis])


data = np.random.normal(0, 1, (100, 100))
data_2 = np.random.normal(1, 0.3, (100, 100))
data_3d = np.append(
    np.reshape(data, (1,) + data.shape), np.reshape(data, (1,) + data.shape), axis=0
)
data_3d = np.append(data_3d, np.reshape(data, (1,) + data.shape), axis=0)

data_3d_3 = np.append(
    np.reshape(data_2, (1,) + data_2.shape),
    np.reshape(data_2, (1,) + data_2.shape),
    axis=0,
)
data_3d_3 = np.append(data_3d_3, np.reshape(data_2, (1,) + data_2.shape), axis=0)
nd_map.data = data_3d
nd_map_2.data = data_3d_3
axes = plot_distribution(
    nd_map,
    density=True,
    bins=20,
    yscale="linear",
    xlim=(-7, 7),
    label="toto",
    title="tata",
    xlabel="sigma",
)
plot_distribution(
    nd_map_2,
    ax=axes,
    density=True,
    bins=20,
    yscale="linear",
    xlim=(-7, 7),
    label="titi",
    title="tata",
    xlabel="sigma",
)

plt.show()

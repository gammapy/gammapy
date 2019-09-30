import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from gammapy.cube import MapDataset, make_map_exposure_true_energy
from gammapy.irf import EffectiveAreaTable2D
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)

filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")

# Define sky model to simulate the data
lon_0_1 = 0.2
lon_0_2 = 0.4
lat_0_1 = 0.1
lat_0_2 = 0.6

spatial_model_1 = GaussianSpatialModel(
    lon_0=lon_0_1 * u.deg, lat_0=lat_0_1 * u.deg, sigma="0.3 deg", frame="galactic"
)
spatial_model_2 = GaussianSpatialModel(
    lon_0=lon_0_2 * u.deg, lat_0=lat_0_2 * u.deg, sigma="0.2 deg", frame="galactic"
)

spectral_model_1 = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)

spectral_model_2 = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)

sky_model_1 = SkyModel(
    spatial_model=spatial_model_1, spectral_model=spectral_model_1, name="source-1"
)

sky_model_2 = SkyModel(
    spatial_model=spatial_model_2, spectral_model=spectral_model_2, name="source-2"
)

models = sky_model_1 + sky_model_2

# Define map geometry
axis = MapAxis.from_edges(np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy")
geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.02, width=(2, 2), coordsys="GAL", axes=[axis]
)


# Define some observation parameters
# we are not simulating many pointings / observations
pointing = SkyCoord(0.2, 0.5, unit="deg", frame="galactic")
livetime = 20 * u.hour

exposure_map = make_map_exposure_true_energy(
    pointing=pointing, livetime=livetime, aeff=aeff, geom=geom
)

dataset = MapDataset(model=models, exposure=exposure_map)
npred = dataset.npred()

fig, ax, cbar = npred.sum_over_axes().plot(add_cbar=True)
ax.scatter(
    [lon_0_1, lon_0_2, pointing.galactic.l.degree],
    [lat_0_1, lat_0_2, pointing.galactic.b.degree],
    transform=ax.get_transform("galactic"),
    marker="+",
    color="cyan",
)
# plt.show()
plt.clf()

dataset.fake()

dataset.counts.sum_over_axes().plot()
# plt.show()
plt.clf()

models.parameters.set_error(spatial_model_1.lon_0, 0.1 * u.deg)
models.parameters.set_error(spatial_model_1.lat_0, 0.1 * u.deg)

models.parameters.set_error(spatial_model_2.lon_0, 0.1 * u.deg)
models.parameters.set_error(spatial_model_1.lat_0, 0.1 * u.deg)

models.parameters.set_error(
    spectral_model_1.amplitude, 1e-12 * u.Unit("cm-2 s-1 TeV-1")
)
models.parameters.set_error(
    spectral_model_2.amplitude, 1e-12 * u.Unit("cm-2 s-1 TeV-1")
)

fit = Fit(dataset)
fit.run()

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import EffectiveAreaTable2D
from gammapy.maps import WcsGeom, MapAxis, WcsNDMap
from gammapy.spectrum.models import PowerLaw
from gammapy.image.models import SkyGaussian
from gammapy.utils.random import get_random_state
from gammapy.cube import make_map_exposure_true_energy, MapFit, MapEvaluator
from gammapy.cube.models import SkyModel

filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")

# Define sky model to simulate the data
lon_0_1 = 0.2
lon_0_2 = 0.4
lat_0_1 = 0.1
lat_0_2 = 0.6

spatial_model_1 = SkyGaussian(
    lon_0=lon_0_1 * u.deg, lat_0=lat_0_1 * u.deg, sigma="0.3 deg"
)
spatial_model_2 = SkyGaussian(
    lon_0=lon_0_2 * u.deg, lat_0=lat_0_2 * u.deg, sigma="0.2 deg"
)

spectral_model_1 = PowerLaw(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)

spectral_model_2 = PowerLaw(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)

sky_model_1 = SkyModel(spatial_model=spatial_model_1, spectral_model=spectral_model_1)

sky_model_2 = SkyModel(spatial_model=spatial_model_2, spectral_model=spectral_model_2)

compound_model = sky_model_1 + sky_model_2

# Define map geometry
axis = MapAxis.from_edges(np.logspace(-1., 1., 10), unit="TeV")
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

evaluator = MapEvaluator(model=compound_model, exposure=exposure_map)


npred = evaluator.compute_npred()
npred_map = WcsNDMap(geom, npred)

fig, ax, cbar = npred_map.sum_over_axes().plot(add_cbar=True)
ax.scatter(
    [lon_0_1, lon_0_2, pointing.galactic.l.degree],
    [lat_0_1, lat_0_2, pointing.galactic.b.degree],
    transform=ax.get_transform("galactic"),
    marker="+",
    color="cyan",
)
# plt.show()
plt.clf()

rng = get_random_state(42)
counts = rng.poisson(npred)
counts_map = WcsNDMap(geom, counts)

counts_map.sum_over_axes().plot()
# plt.show()
plt.clf()

compound_model.parameters.set_error(2, 0.1 * u.deg)
compound_model.parameters.set_error(4, 1e-12 * u.Unit("cm-2 s-1 TeV-1"))
compound_model.parameters.set_error(8, 0.1 * u.deg)
compound_model.parameters.set_error(10, 1e-12 * u.Unit("cm-2 s-1 TeV-1"))

fit = MapFit(model=compound_model, counts=counts_map, exposure=exposure_map)
fit.run()

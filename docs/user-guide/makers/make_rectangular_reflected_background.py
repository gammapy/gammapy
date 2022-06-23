"""Example how to compute and plot reflected regions."""
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset
from gammapy.makers import ReflectedRegionsBackgroundMaker, SpectrumDatasetMaker
from gammapy.maps import Map, MapAxis, RegionGeom
from gammapy.visualization import plot_spectrum_datasets_off_regions

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
mask = data_store.obs_table["TARGET_NAME"] == "Crab"
obs_ids = data_store.obs_table["OBS_ID"][mask].data
observations = data_store.get_observations(obs_ids)

crab_position = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")

# The ON region center is defined in the icrs frame. The angle is defined w.r.t. to its axis.
rectangle = RectangleSkyRegion(
    center=crab_position, width=0.5 * u.deg, height=0.4 * u.deg, angle=0 * u.deg
)

bkg_maker = ReflectedRegionsBackgroundMaker(min_distance=0.1 * u.rad)
dataset_maker = SpectrumDatasetMaker(selection=["counts"])

energy_axis = MapAxis.from_energy_bounds(0.1, 100, 30, unit="TeV")
geom = RegionGeom.create(region=rectangle, axes=[energy_axis])
dataset_empty = SpectrumDataset.create(geom=geom)

datasets = []

for obs in observations:

    dataset = dataset_maker.run(dataset_empty.copy(name=f"obs-{obs.obs_id}"), obs)
    dataset_on_off = bkg_maker.run(observation=obs, dataset=dataset)
    datasets.append(dataset_on_off)

m = Map.create(skydir=crab_position, width=(8, 8), proj="TAN")

ax = m.plot(vmin=-1, vmax=0)

rectangle.to_pixel(ax.wcs).plot(ax=ax, color="black")

plot_spectrum_datasets_off_regions(datasets=datasets, ax=ax)
plt.show()

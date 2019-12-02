"""Example how to compute and plot reflected regions."""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.maps import Map
from gammapy.spectrum import (
    ReflectedRegionsBackgroundMaker,
    SpectrumDataset,
    SpectrumDatasetMaker,
    plot_spectrum_datasets_off_regions,
)

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

dataset_empty = SpectrumDataset.create(
    e_reco=np.logspace(-1, 2, 30) * u.TeV, region=rectangle
)

datasets = []

for obs in observations:
    dataset = dataset_maker.run(dataset_empty, obs)
    dataset_on_off = bkg_maker.run(observation=obs, dataset=dataset)
    datasets.append(dataset_on_off)

m = Map.create(skydir=crab_position, width=(8, 8), proj="TAN")

_, ax, _ = m.plot(vmin=-1, vmax=0)

rectangle.to_pixel(ax.wcs).plot(ax=ax, color="black")

plot_spectrum_datasets_off_regions(datasets=datasets, ax=ax)
plt.show()

"""
Energy Dependence Estimation
============================

Learn how to test for energy dependent morphology in your dataset.

Prerequisites
-------------
Knowledge on data reduction and datasets used in Gammapy, for example see
the :doc:`/tutorials/data/hess` and :doc:`/tutorials/analysis-2d/ring_background` tutorials.


Context
-------

A tool to investigate the potential of energy-dependent morphology from spatial maps. This tutorial consists
of two main steps.

Firstly, the user defines both the initial SkyModel as determined through previous investigate along with the
energy-band of interest to test for energy dependence. The null hypothesis (H0) is defined as only the background
component being free (norm). The alternative hypothesis (H1) is adding the source model back in. The results of this
first step show the significance of your source above the background in each energy band.

The second step is for the purpose of quantifying any energy dependent morphology. The null hypothesis here is defined
through a joint fit of parameters. Giving a base for the analysis. The alternative hypothesis is where the free
parameters of the model are fit individually in each energy band.

Setup
-----

"""

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import Datasets, MapDataset
from gammapy.estimators.energydependence import EnergyDependenceEstimator
from gammapy.makers import (
    DatasetsMaker,
    FoVBackgroundMaker,
    MapDatasetMaker,
    SafeMaskMaker,
)
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.stats.utils import ts_to_sigma

######################################################################
# Check setup
# -----------

# from gammapy.utils.check import check_tutorials_setup
# check_tutorials_setup()

######################################################################
# Obtain the data to use
# ----------------------
#
# Create the data store and obtain the observations from the HESS DL3 DR1 for MSH 1552.
#
# P.S.: do not forget to setup your environment variable $GAMMAPY_DATA to your local directory

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
obs_id = data_store.obs_table["OBS_ID"][data_store.obs_table["OBJECT"] == "MSH 15-5-02"]
observations = data_store.get_observations(obs_id)

######################################################################
# Setting the exclusion mask
# --------------------------
#
# First, we define the energy range to obtain the dataset within. The geometry is also
# defined, based on the position of MSH 1552 (the source of interest here).

energy_axis = MapAxis.from_energy_bounds(0.2, 100, nbin=15, unit="TeV")
energy_axis_true = MapAxis.from_energy_bounds(
    0.05, 110, nbin=30, unit="TeV", name="energy_true"
)

source_pos = SkyCoord(320.33, -1.19, unit="deg", frame="galactic")
geom = WcsGeom.create(
    skydir=(source_pos.galactic.l.deg, source_pos.galactic.b.deg),
    frame="galactic",
    axes=[energy_axis],
    width=5,
    binsz=0.02,
)
regions = CircleSkyRegion(center=source_pos, radius=0.7 * u.deg)
exclusion_mask = geom.region_mask(regions, inside=False)
exclusion_mask.sum_over_axes().plot()

######################################################################
# Data reduction loop
# -------------------
#
# For details on how the data reduction is performed see the
# :doc:`/tutorials/analysis-3d/analysis_3d` tutorial.
#
# The data reduction steps can be combined using the DatasetsMaker class
# which takes as an input the list of makers.
# Here we stack the dataset in this step.
#

safe_mask_maker = SafeMaskMaker(
    methods=["aeff-default", "offset-max"], offset_max=2.5 * u.deg
)

dataset_maker = MapDatasetMaker()

fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

global_dataset = MapDataset.create(geom)
makers = [dataset_maker, safe_mask_maker, fov_bkg_maker]  # the order matters

datasets_maker = DatasetsMaker(
    makers, stack_datasets=True, n_jobs=1, cutout_mode="partial"
)

datasets = datasets_maker.run(global_dataset, observations)

######################################################################
# Define the energy edges of interest. These will be utilised to
# investigate the potential of energy-dependent morphology in the dataset.

energy_edges = [0.3, 1, 5, 10] * u.TeV

######################################################################
# Define the spectral and spatial models of interest. Here we utilise
# a PowerLawSpectralModel and a GaussianSpatialModel to test the energy
# dependent morphology component in each energy band. A standard 3D fit
# is performed then the best fit model is utilised here for the parameters
# in each model.

spectral_model = PowerLawSpectralModel(
    index=2.26, amplitude=2.58e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1.0 * u.TeV
)

spatial_model = GaussianSpatialModel(
    lon_0=source_pos.l,
    lat_0=source_pos.b,
    frame="galactic",
    sigma=0.11 * u.deg,
    e=0.8346,
    phi=-2.914 * u.deg,
)

# Limit the search for the position on the spatial model
spatial_model.lon_0.min = source_pos.galactic.l.deg - 0.8
spatial_model.lon_0.max = source_pos.galactic.l.deg + 0.8
spatial_model.lat_0.min = source_pos.galactic.b.deg - 0.8
spatial_model.lat_0.max = source_pos.galactic.b.deg + 0.8

model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="MSH1552"
)

######################################################################
# Run Estimator
# -------------
#
# Now we can run the energy dependent estimation and explore the results.
#
# We start with the initial hypothesis, in which the source is added
# in to compare with the background. We define here which parameters we
# wish to use to test the energy dependence. To test the energy dependence
# it is advised to let the position and extension parameter to be free
# such that these can be used to fit the spatial model in each band.

model.spatial_model.lon_0.frozen = False
model.spatial_model.lat_0.frozen = False
model.spatial_model.sigma.frozen = False

model.spectral_model.amplitude.frozen = False
model.spectral_model.index.frozen = True

datasets.model = model

estimator = EnergyDependenceEstimator(energy_edges=energy_edges, source="MSH1552")

######################################################################
# Show the results tables
# -----------------------
#
# The results for the source above the background
# -----------------------------------------------
#

estimator = EnergyDependenceEstimator(
    energy_edges=energy_edges, source=source, selection_optional=["src-sig"]
)
table_bkg_src = estimator.estimate_source_significance(datasets)
table_bkg_src

######################################################################
# The results for testing energy dependence
# -----------------------------------------

results = estimator.run(datasets)
ts = results["energy_dependence"]["delta_ts"]
df = results["energy_dependence"]["df"]
print(f"The significance value is {ts_to_sigma(ts, df=df)}")
table = Table(results["energy_dependence"]["result"])
table

######################################################################
# Plotting the results
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
from gammapy.maps import Map

empty_map = Map.create(
    skydir=spatial_model.position, frame=spatial_model.frame, width=0.7, binsz=0.02
)

colors = ["red", "blue", "green", "magenta"]

fig = plt.figure(figsize=(6, 4))
ax = empty_map.plot()

lat_0 = results["energy_dependence"]["result"]["lat_0"][1:]
lat_0_err = results["energy_dependence"]["result"]["lat_0_err"][1:]
lon_0 = results["energy_dependence"]["result"]["lon_0"][1:]
lon_0_err = results["energy_dependence"]["result"]["lon_0_err"][1:]
sigma = results["energy_dependence"]["result"]["sigma"][1:]
sigma_err = results["energy_dependence"]["result"]["sigma_err"][1:]

for i in range(len(lat_0)):
    model_plot = GaussianSpatialModel(
        lat_0=lat_0[i], lon_0=lon_0[i], sigma=sigma[i], frame=spatial_model.frame
    )
    model_plot.lat_0.error = lat_0_err[i]
    model_plot.lon_0.error = lon_0_err[i]
    model_plot.sigma.error = sigma_err[i]

    model_plot.plot_error(
        ax=ax,
        which="all",
        kwargs_extension={"facecolor": colors[i], "edgecolor": colors[i]},
        kwargs_position={"color": colors[i]},
    )

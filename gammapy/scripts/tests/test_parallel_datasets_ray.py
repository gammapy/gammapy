import time
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import ray
from gammapy.data import Observation
from gammapy.data.pointing import FixedPointingInfo, PointingMode
from gammapy.datasets import Datasets, DatasetsActor, MapDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    GaussianSpatialModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

ray.shutdown()
ray.init(num_cpus=3, num_gpus=1)
# ray.shutdown()


# Loading IRFs
irfs = load_irf_dict_from_file(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

# %% prepare simulation

# Define the observation parameters (typically the observation duration and the pointing position):
livetime = 2.0 * u.hr
lons = [-1, 0, 1]
pointings = []
for lon in lons:
    pointings.append(
        FixedPointingInfo(
            mode=PointingMode.POINTING,
            fixed_icrs=SkyCoord(lon, 0, unit="deg", frame="galactic").icrs,
        )
    )


# Define map geometry for binned simulation
energy_reco = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(6, 6),
    frame="galactic",
    axes=[energy_reco],
)
# It is usually useful to have a separate binning for the true energy axis
energy_true = MapAxis.from_edges(
    np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy", interp="log"
)


# Define sky model to used simulate the data.
# Here we use a Gaussian spatial model and a Power Law spectral model.
spatial_model = GaussianSpatialModel(
    lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
)
spectral_model = PowerLawSpectralModel(
    index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_simu = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="model-simu",
)
models_simu = Models([model_simu])

# prepare makers
maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

# %% Create datasets with loop
start_time = time.time()
datasets = Datasets()
for idx, pointing in enumerate(pointings):
    empty = MapDataset.create(geom, name=f"obs_{idx}")
    obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
    dataset = maker.run(empty, obs)
    dataset = maker_safe_mask.run(dataset, obs)
    dataset.models = models_simu
    dataset.fake()
    dataset.models = None
    datasets.append(dataset)
exec_time = time.time() - start_time
print("\n fake: time in seconds: ", exec_time)

# %% Define sky model to fit the data
spatial_model1 = GaussianSpatialModel(
    lon_0="0.1 deg", lat_0="0.1 deg", sigma="0.5 deg", frame="galactic"
)
spectral_model1 = PowerLawSpectralModel(
    index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_fit = SkyModel(
    spatial_model=spatial_model1,
    spectral_model=spectral_model1,
    name="model-fit",
)
models_fit = Models([model_fit])

datasets.models = models_fit
print("\n", datasets.models)
init_values = datasets.parameters.get_parameter_values()


# %%

ncall = 300
start_time = time.time()
datasets.models["model-fit"].spectral_model.index.value = 2
for k in range(ncall):
    datasets.models["model-fit"].spectral_model.index.value += 1 / (k + 1)
    res = datasets.stat_sum()
    if k == 0:
        print(res)
exec_time = time.time() - start_time
print(f"\n {ncall} stat_sum: time in seconds: ", exec_time)


start_time = time.time()
da = DatasetsActor(datasets)
da.models["model-fit"].spectral_model.index.value = 2
for k in range(ncall):
    da.models["model-fit"].spectral_model.index.value += 1 / (k + 1)
    res = da.stat_sum()
    if k == 0:
        print(res)
exec_time = time.time() - start_time
print(f"\n {ncall} stat_sum: ray time in seconds: ", exec_time)

# %% fit

start_time = time.time()
da.parameters.set_parameter_values(init_values)
fit = Fit(optimize_opts={"print_level": 1})
result = fit.run(da)
exec_time = time.time() - start_time
print("\n ray Fit time in seconds: ", exec_time)
print("ray fit parameters:", da.parameters.get_parameter_values())

# result.parameters.to_table()

# %% model management

# Create datasets with ray
start_time = time.time()


@ray.remote
def fake_dataset(idx, pointing):
    empty = MapDataset.create(geom, name=f"obs_{idx}")
    obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
    dataset = maker.run(empty, obs)
    dataset = maker_safe_mask.run(dataset, obs)
    dataset.models = models_simu
    dataset.fake()
    dataset.models = None
    return dataset


datasets_list = ray.get([fake_dataset.remote(k, p) for k, p in enumerate(pointings)])
exec_time = time.time() - start_time
print("\n fake: ray time in seconds: ", exec_time)

# Define sky model to fit the data
spatial_model1 = GaussianSpatialModel(
    lon_0="0.1 deg", lat_0="0.1 deg", sigma="0.5 deg", frame="galactic"
)
spectral_model1 = PowerLawSpectralModel(
    index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_fit = SkyModel(
    spatial_model=spatial_model1,
    spectral_model=spectral_model1,
    name="model-fit",
)

start_time = time.time()
da = DatasetsActor(datasets_list)
# append  model on each (ghost) dataset but their remote actor is not updated yet
da.models = models_fit

fit = Fit(optimize_opts={"print_level": 1})
# remote actors updated only on fit.run()
result = fit.run(da)
exec_time = time.time() - start_time
print("\n ray Fit time in seconds: ", exec_time)
print("ray fit parameters:", da.parameters.get_parameter_values())

da.models[0].spectral_model.amplitude.value = 1
da.plot_residuals()
da.plot_residuals(update_remote=True)

# ray.shutdow()

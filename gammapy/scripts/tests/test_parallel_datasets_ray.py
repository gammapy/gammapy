import ray
import time
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import load_cta_irfs
from gammapy.maps import WcsGeom, MapAxis
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    GaussianSpatialModel,
    SkyModel,
)
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.modeling import Fit
from gammapy.data import Observation
from gammapy.datasets import MapDataset, Datasets, MapDatasetActor

ray.shutdown()
ray.init(num_cpus=3, num_gpus=1)
# ray.shutdown()


# Loading IRFs
irfs = load_cta_irfs(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

#%%
# prepare simulation

# Define the observation parameters (typically the observation duration and the pointing position):
livetime = 2.0 * u.hr
lobs = [-1, 0, 1]
pointings = [SkyCoord(l, 0, unit="deg", frame="galactic") for l in lobs]


# Define map geometry for binned simulation
energy_reco = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.02, width=(6, 6), frame="galactic", axes=[energy_reco],
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
    spatial_model=spatial_model, spectral_model=spectral_model, name="model-simu",
)

# prepare makers
maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

#%%
# Create datasets with loop
start_time = time.time()
datasets_list = []
for idx, pointing in enumerate(pointings):
    empty = MapDataset.create(geom, name=f"obs_{idx}")
    obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
    dataset = maker.run(empty, obs)
    dataset = maker_safe_mask.run(dataset, obs)
    dataset.models.append(model_simu)
    dataset.fake()
    dataset.models.remove(model_simu)
    datasets_list.append(dataset)
exec_time = time.time() - start_time
print("\n fake: time in seconds: ", exec_time)

#%%
## Create datasets with ray
# start_time = time.time()
# @ray.remote
# def fake_dataset(idx, pointing):
#    empty = MapDataset.create(geom, name=f"obs_{idx}")
#    obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
#    dataset = maker.run(empty, obs)
#    dataset = maker_safe_mask.run(dataset, obs)
#    dataset.models.append(model_simu)
#    dataset.fake()
#    dataset.models.remove(model_simu)
#    print(idx, pointing)
#    return dataset
# datasets_list = ray.get([fake_dataset.remote(k, p) for  k , p in enumerate(pointings)])
# exec_time = time.time() - start_time
# print("\n fake: ray time in seconds: ", exec_time)
#

# This a bit is faster  but cause fail afterward in fit with the following error:
#  File "/Users/qremy/Work/GitHub/gammapy/gammapy/modeling/covariance.py", line 139, in set_subcovariance
#    self._data[np.ix_(idx, idx)] = covar.data
#
# ValueError: assignment destination is read-only

#%%

# Define sky model to fit the data
spatial_model1 = GaussianSpatialModel(
    lon_0="0.1 deg", lat_0="0.1 deg", sigma="0.5 deg", frame="galactic"
)
spectral_model1 = PowerLawSpectralModel(
    index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model_fit = SkyModel(
    spatial_model=spatial_model1, spectral_model=spectral_model1, name="model-fit",
)

datasets = Datasets(datasets_list)
datasets.models.append(model_fit)
print("\n", datasets.models)

init_values = datasets.parameters.get_parameter_values()

#%%

ncall = 300
start_time = time.time()
datasets.models["model-fit"].spectral_model.index.value = 2
for k in range(ncall):
    datasets.models["model-fit"].spectral_model.index.value += 1 / (k + 1)
    res = [d.stat_sum() for d in datasets]
    if k == 0:
        print(res)
exec_time = time.time() - start_time
print(f"\n {ncall} stat_sum: time in seconds: ", exec_time)


start_time = time.time()
actors = [MapDatasetActor.remote(d) for d in datasets]
datasets.models["model-fit"].spectral_model.index.value = 2
for k in range(ncall):
    datasets.models["model-fit"].spectral_model.index.value += 1 / (k + 1)
    args = [d.models.parameters.get_parameter_values() for d in datasets]
    ray.get([a.set_parameter_values.remote(arg) for a, arg in zip(actors, args)])
    # blocked until set_parameters_factors on actors complete
    res = ray.get([a.stat_sum.remote() for a in actors])
    if k == 0:
        print(res)
exec_time = time.time() - start_time
print(f"\n {ncall} stat_sum: ray time in seconds: ", exec_time)


#%%
# fit

start_time = time.time()
datasets.parameters.set_parameter_values(init_values)
fit = Fit(datasets)
result = fit.run(optimize_opts={"print_level": 1})
exec_time = time.time() - start_time
print("\n Fit time in seconds: ", exec_time)
print("fit paraemters:", datasets.parameters.get_parameter_values())

start_time = time.time()
datasets.parameters.set_parameter_values(init_values)
fit = Fit(datasets, parallel=True)
result = fit.run(optimize_opts={"print_level": 1})
exec_time = time.time() - start_time
print("\n ray Fit time in seconds: ", exec_time)
print("ray fit parameters:", datasets.parameters.get_parameter_values())

# result.parameters.to_table()

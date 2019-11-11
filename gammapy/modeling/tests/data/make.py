"""Create example model YAML files programmatically.

(some will be also written manually)
"""
from pathlib import Path
import numpy as np
import astropy.units as u
from gammapy.cube import MapDataset, MapDatasetMaker
from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Datasets
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    GaussianSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyDiffuseCube,
    SkyModel,
    SkyModels,
)

DATA_PATH = Path("./")


def make_example_2():
    spatial = GaussianSpatialModel("0 deg", "0 deg", "1 deg")
    model = SkyModel(spatial, PowerLawSpectralModel())
    models = SkyModels([model])
    models.to_yaml(DATA_PATH / "example2.yaml")


def make_datasets_example():
    # Define which data to use and print some information

    energy_axis = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 4), unit="TeV", name="energy", interp="log"
    )
    geom0 = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.1,
        width=(1, 1),
        coordsys="GAL",
        proj="CAR",
        axes=[energy_axis],
    )
    geom1 = WcsGeom.create(
        skydir=(1, 0),
        binsz=0.1,
        width=(1, 1),
        coordsys="GAL",
        proj="CAR",
        axes=[energy_axis],
    )
    geoms = [geom0, geom1]

    sources_coords = [(0, 0), (0.9, 0.1)]
    names = ["gc", "g09"]
    models = []

    for idx, (lon, lat) in enumerate(sources_coords):
        spatial_model = PointSpatialModel(
            lon_0=lon * u.deg, lat_0=lat * u.deg, frame="galactic"
        )
        spectral_model = ExpCutoffPowerLawSpectralModel(
            index=2 * u.Unit(""),
            amplitude=3e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1.0 * u.TeV,
            lambda_=0.1 / u.TeV,
        )
        model_ecpl = SkyModel(
            spatial_model=spatial_model, spectral_model=spectral_model, name=names[idx]
        )
        models.append(model_ecpl)

    # test to link a spectral parameter
    params0 = models[0].spectral_model.parameters
    params1 = models[1].spectral_model.parameters
    params0.link("reference", params1["reference"])
    # update the sky model
    models[0].parameters.link("reference", params1["reference"])

    obs_ids = [110380, 111140, 111159]
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")

    diffuse_model = SkyDiffuseCube.read(
        "$GAMMAPY_DATA/fermi_3fhl/gll_iem_v06_cutout.fits"
    )

    datasets_list = []
    for idx, geom in enumerate(geoms):
        observations = data_store.get_observations(obs_ids)

        stacked = MapDataset.create(geom=geom)
        stacked.background_model.name = "background_irf_" + names[idx]

        maker = MapDatasetMaker(geom=geom, offset_max=4.0 * u.deg)

        for obs in observations:
            dataset = maker.run(obs)
            stacked.stack(dataset)

        stacked.psf = stacked.psf.get_psf_kernel(
            position=geom.center_skydir, geom=geom, max_radius="0.3 deg"
        )
        stacked.edisp = stacked.edisp.get_energy_dispersion(
            position=geom.center_skydir, e_reco=energy_axis.edges
        )

        stacked.name = names[idx]
        stacked.model = models[idx] + diffuse_model
        datasets_list.append(stacked)

    datasets = Datasets(datasets_list)

    dataset0 = datasets[0]
    print("dataset0")
    print("counts sum : ", dataset0.counts.data.sum())
    print("expo sum : ", dataset0.exposure.data.sum())
    print("bkg0 sum : ", dataset0.background_model.evaluate().data.sum())

    datasets.to_yaml("$GAMMAPY_DATA/tests/models", prefix="gc_example_", overwrite=True)


if __name__ == "__main__":
    make_example_2()
    make_datasets_example()

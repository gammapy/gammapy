"""Create example model YAML files programmatically.

(some will be also written manually)
"""
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.cube import MapDataset, MapEvaluator, MapMaker, PSFKernel
from gammapy.data import DataStore
from gammapy.irf import make_mean_edisp, make_mean_psf
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Datasets
from gammapy.modeling.models import (
    BackgroundModel,
    ExponentialCutoffPowerLaw,
    PowerLaw,
    SkyDiffuseCube,
    SkyGaussian,
    SkyModel,
    SkyModels,
    SkyPointSource,
)

DATA_PATH = Path("./")


def make_example_2():
    spatial = SkyGaussian("0 deg", "0 deg", "1 deg")
    model = SkyModel(spatial, PowerLaw())
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

    for ind, (lon, lat) in enumerate(sources_coords):
        spatial_model = SkyPointSource(lon_0=lon * u.deg, lat_0=lat * u.deg)
        spectral_model = ExponentialCutoffPowerLaw(
            index=2 * u.Unit(""),
            amplitude=3e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1.0 * u.TeV,
            lambda_=0.1 / u.TeV,
        )
        model_ecpl = SkyModel(
            spatial_model=spatial_model, spectral_model=spectral_model, name=names[ind]
        )
        models.append(model_ecpl)

    # test to link a spectral parameter
    params0 = models[0].spectral_model.parameters
    params1 = models[1].spectral_model.parameters
    ind = params0.parameters.index(params0["reference"])
    params0.parameters[ind] = params1["reference"]

    # update the sky model
    ind = models[0].parameters.parameters.index(models[0].parameters["reference"])
    models[0].parameters.parameters[ind] = params1["reference"]

    obs_ids = [110380, 111140, 111159]
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")

    diffuse_model = SkyDiffuseCube.read(
        "$GAMMAPY_DATA/fermi_3fhl/gll_iem_v06_cutout.fits"
    )

    datasets_list = []
    for ind, geom in enumerate(geoms):
        observations = data_store.get_observations(obs_ids)

        maker = MapMaker(geom, offset_max=4.0 * u.deg)
        maps = maker.run(observations)

        src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
        table_psf = make_mean_psf(observations, src_pos)
        psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius="0.3 deg")

        energy = energy_axis.edges
        edisp = make_mean_edisp(
            observations, position=src_pos, e_true=energy, e_reco=energy
        )

        background_irf = BackgroundModel(
            maps["background"], norm=1.0, tilt=0.0, name="background_irf_" + names[ind]
        )

        dataset = MapDataset(
            name=names[ind],
            model=models[ind] + diffuse_model,
            counts=maps["counts"],
            exposure=maps["exposure"],
            background_model=background_irf,
            psf=psf_kernel,
            edisp=edisp,
        )
        datasets_list.append(dataset)

    datasets = Datasets(datasets_list)

    dataset0 = datasets.datasets[0]
    print("dataset0")
    print("counts sum : ", dataset0.counts.data.sum())
    print("expo sum : ", dataset0.exposure.data.sum())
    print("bkg0 sum : ", dataset0.background_model.evaluate().data.sum())

    path = "$GAMMAPY_DATA/tests/models/gc_example_"
    datasets.to_yaml(path, selection="simple", overwrite=True)


if __name__ == "__main__":
    make_example_2()
    make_datasets_example()

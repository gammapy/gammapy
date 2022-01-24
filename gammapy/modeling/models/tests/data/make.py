"""Create example model YAML files programmatically.

(some will be also written manually)
"""
from pathlib import Path
import numpy as np
import astropy.units as u
from gammapy.data import DataStore
from gammapy.datasets import Datasets, MapDataset
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    GaussianSpatialModel,
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)

DATA_PATH = Path("./")


def make_example_2():
    spatial = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="1 deg")
    model = SkyModel(PowerLawSpectralModel(), spatial, name="example_2")
    models = Models([model])
    models.write(DATA_PATH / "example2.yaml", overwrite=True, write_covariance=False)


def make_datasets_example():
    # Define which data to use and print some information

    energy_axis = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 4), unit="TeV", name="energy", interp="log"
    )
    geom0 = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.1,
        width=(2, 2),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis],
    )
    geom1 = WcsGeom.create(
        skydir=(1, 0),
        binsz=0.1,
        width=(2, 2),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis],
    )
    geoms = [geom0, geom1]

    sources_coords = [(0, 0), (0.9, 0.1)]
    names = ["gc", "g09"]
    models = Models()

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

    models["gc"].spectral_model.reference = models["g09"].spectral_model.reference

    obs_ids = [110380, 111140, 111159]
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")

    diffuse_spatial = TemplateSpatialModel.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
    )
    diffuse_model = SkyModel(PowerLawSpectralModel(), diffuse_spatial)

    maker = MapDatasetMaker()
    datasets = Datasets()

    observations = data_store.get_observations(obs_ids)

    for idx, geom in enumerate(geoms):
        stacked = MapDataset.create(geom=geom, name=names[idx])

        for obs in observations:
            dataset = maker.run(stacked, obs)
            stacked.stack(dataset)

        bkg = stacked.models.pop(0)
        stacked.models = [models[idx], diffuse_model, bkg]
        datasets.append(stacked)

    datasets.write(
        filename="$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml",
        filename_models="$GAMMAPY_DATA/tests/models/gc_example_models.yaml",
        overwrite=True,
        write_covariance=False,
    )


if __name__ == "__main__":
    make_example_2()
    make_datasets_example()

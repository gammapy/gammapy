# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.cube import MapDataset, MapDatasetEventSampler, simulate_dataset
from gammapy.cube.tests.test_fit import get_map_dataset
from gammapy.data import GTI
from gammapy.irf import load_cta_irfs
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    SkyModels,
)
from gammapy.utils.testing import requires_data


@requires_data()
def test_simulate():
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    # Define sky model to simulate the data
    spatial_model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    sky_model_simu = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model
    )

    # Define map geometry
    axis = MapAxis.from_edges(
        np.logspace(-1, 1.0, 20), unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.025, width=(1, 1), coordsys="GAL", axes=[axis]
    )

    # Define some observation parameters
    pointing = SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic")

    dataset = simulate_dataset(
        sky_model_simu, geom, pointing, irfs, livetime=10 * u.h, random_state=42
    )

    assert isinstance(dataset, MapDataset)
    assert isinstance(dataset.models, SkyModels)

    assert dataset.counts.data.dtype is np.dtype("int")
    assert_allclose(dataset.counts.data[5, 20, 20], 2)
    assert_allclose(dataset.exposure.data[5, 20, 20], 16122681486.381285)
    assert_allclose(
        dataset.background_model.map.data[5, 20, 20], 0.9765545345855245, rtol=1e-5
    )
    assert_allclose(dataset.psf.psf_map.data[5, 5, 0, 0], 91987.862)
    assert_allclose(dataset.edisp.data.data[10, 10], 0.85944298, rtol=1e-5)


def dataset_maker():
    position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
    energy_axis = MapAxis.from_bounds(
        1, 100, nbin=30, unit="TeV", name="energy", interp="log"
    )

    spatial_model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
    )

    spectral_model = PowerLawSpectralModel(amplitude="1e-11 cm-2 s-1 TeV-1")
    skymodel = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    geom = WcsGeom.create(
        skydir=position, binsz=0.02, width="5 deg", coordsys="GAL", axes=[energy_axis]
    )

    t_min = 0 * u.s
    t_max = 30000 * u.s

    gti = GTI.create(start=t_min, stop=t_max)

    dataset = get_map_dataset(
        sky_model=skymodel, geom=geom, geom_etrue=geom, edisp=True
    )
    dataset.gti = gti

    return dataset


@requires_data()
def test_MDE_sample_background():
    dataset = dataset_maker()
    sampler = MapDatasetEventSampler(random_state=0)
    bkg_evt = sampler.sample_background(dataset=dataset)

    assert len(bkg_evt.table["ENERGY"]) == 375084
    assert_allclose(bkg_evt.table["ENERGY"][0], 2.1613281656472028, rtol=1e-5)
    assert_allclose(bkg_evt.table["RA"][0], 265.7253792887848, rtol=1e-5)
    assert_allclose(bkg_evt.table["DEC"][0], -27.727581635186304, rtol=1e-5)
    assert_allclose(bkg_evt.table["MC_ID"][0], 0, rtol=1e-5)

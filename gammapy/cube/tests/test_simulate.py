# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.cube import MapDataset, simulate_map
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
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.3 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=2.0, amplitude="1e-10 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model_simu = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    # Define map geometry
    energy_reco = MapAxis.from_edges(
        np.logspace(-1, 1.0, 10), unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.02, width=(2, 2), coordsys="GAL", axes=[energy_reco]
    )
    energy_axis_true = MapAxis.from_nodes(
        np.logspace(0.1, 2, 20), unit="TeV", name="energy_axis_true", interp="log"
    )

    # Define some observation parameters
    pointing = SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic")

    livetime = 10 * u.h

    dataset = simulate_map(
        skymodel=model_simu,
        geom=geom,
        pointing=pointing,
        irfs=irfs,
        livetime=livetime,
        background_pars={"norm": 1.0, "tilt": 0.0},
        random_state=42,
    )

    assert isinstance(dataset, MapDataset)
    assert isinstance(dataset.model, SkyModels)

    assert dataset.counts.data.dtype is np.dtype("int")
    assert_allclose(dataset.counts.data[4, 29, 21], 2)
    assert_allclose(dataset.counts.data.shape, [9, 100, 100])
    assert_allclose(dataset.exposure.data.shape, [9, 100, 100])
    assert_allclose(dataset.exposure.data[5, 20, 20], 5.208063e10)
    assert_allclose(dataset.background_model.map.data[5, 20, 20], 0.162572, rtol=1e-5)
    # assert_allclose(dataset.psf.data[5, 32, 32], 0.04203219)
    assert_allclose(dataset.edisp.edisp_map.data.shape, [9, 48, 12, 12])

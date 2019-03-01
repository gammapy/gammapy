# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from ...irf import load_cta_irfs
from ...maps import WcsGeom, MapAxis
from ...spectrum.models import PowerLaw
from ...image.models import SkyGaussian
from ...cube.models import SkyModel
from ...cube import MapDataset
from ..simulate import simulate_3d


def test_simulate():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    irfs = load_cta_irfs(filename)

    # Define sky model to simulate the data
    spatial_model = SkyGaussian(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg")
    spectral_model = PowerLaw(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    sky_model_simu = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model
    )

    # Define map geometry
    axis = MapAxis.from_edges(
        np.logspace(-1, 1.0, 10), unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.05, width=(1, 1), coordsys="GAL", axes=[axis]
    )

    # Define some observation parameters
    pointing = SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic")

    dataset = simulate_3d(
        sky_model_simu, geom, pointing, irfs, livetime=1 * u.h, edisp=True
    )

    assert isinstance(dataset, MapDataset)
    assert isinstance(dataset.model, SkyModel)

    assert dataset.counts.data.dtype is np.dtype("int")
    assert np.sum(dataset.counts.data) > 0
    assert np.sum(dataset.exposure.data) > 0
    assert np.sum(dataset.edisp.data.data) > 0
    assert np.sum(dataset.psf.data) > 0
    assert np.sum(dataset.background_model.map.data) > 0

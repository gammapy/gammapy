# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import ObservationsEventsSampler
from gammapy.utils.testing import requires_data
from gammapy.irf import load_cta_irfs
from gammapy.data import Observation
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    PointSpatialModel,
    SkyModel,
    Models,
)


@pytest.fixture(scope="session")
def observations():
    pointing = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
    livetime = 0.5 * u.hr
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
    )
    observations = [
        Observation.create(
            obs_id=100 + k, pointing=pointing, livetime=livetime, irfs=irfs
        )
        for k in range(2)
    ]
    return observations


@pytest.fixture(scope="session")
def models():
    spectral_model_pwl = PowerLawSpectralModel(
        index=2, amplitude="1e-12 TeV-1 cm-2 s-1", reference="1 TeV"
    )
    spatial_model_point = PointSpatialModel(
        lon_0="0 deg", lat_0="0.0 deg", frame="galactic"
    )

    sky_model_pntpwl = SkyModel(
        spectral_model=spectral_model_pwl,
        spatial_model=spatial_model_point,
        name="point-pwl",
    )
    models = Models(sky_model_pntpwl)
    return models


@requires_data()
def test_observations_events_sampler(tmpdir, observations):
    sampler = ObservationsEventsSampler(
        models=None,
        outdir=tmpdir,
        prefix="test",
        binsz_min=0.1 * u.deg,
        width_max=0.2 * u.deg,
        nbin_per_decade_max=2,
        n_jobs=1,
        random_state=0,
        overwrite=True,
    )
    sampler.run(observations)


@requires_data()
def test_observations_events_sampler_parralel(tmpdir, observations, models):
    sampler = ObservationsEventsSampler(
        models=models,
        outdir=tmpdir,
        prefix="test",
        binsz_min=0.1 * u.deg,
        width_max=0.2 * u.deg,
        nbin_per_decade_max=2,
        n_jobs=2,
        random_state=0,
        overwrite=True,
    )
    sampler.run(observations)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import FixedPointingInfo, Observation, ObservationsEventsSampler
from gammapy.irf import load_irf_dict_from_file
from gammapy.modeling.models import (
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations():
    pointing = FixedPointingInfo(fixed_icrs=SkyCoord(0 * u.deg, 0 * u.deg))
    livetime = 0.5 * u.hr
    irfs = load_irf_dict_from_file(
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
    sampler_kwargs = dict(random_state=0)
    dataset_kwargs = dict(
        spatial_bin_size_min=0.1 * u.deg,
        spatial_width_max=0.2 * u.deg,
        energy_bin_per_decade_max=2,
    )
    sampler = ObservationsEventsSampler(
        sampler_kwargs=sampler_kwargs,
        dataset_kwargs=dataset_kwargs,
        n_jobs=1,
        outdir=tmpdir,
        overwrite=True,
    )
    sampler.run(observations, models=None)


@requires_data()
def test_observations_events_sampler_parallel(tmpdir, observations, models):
    sampler_kwargs = dict(random_state=0)
    dataset_kwargs = dict(
        spatial_bin_size_min=0.1 * u.deg,
        spatial_width_max=0.2 * u.deg,
        energy_bin_per_decade_max=2,
    )
    sampler = ObservationsEventsSampler(
        sampler_kwargs=sampler_kwargs,
        dataset_kwargs=dataset_kwargs,
        n_jobs=2,
        outdir=tmpdir,
        overwrite=True,
    )
    sampler.run(observations, models=models)

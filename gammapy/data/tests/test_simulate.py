# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.table import Table
import numpy as np
from gammapy.data import Observation, ObservationsEventsSampler
from gammapy.data.pointing import FixedPointingInfo
from gammapy.datasets.tests.test_simulate import get_energy_dependent_temporal_model
from gammapy.irf import load_irf_dict_from_file
from gammapy.modeling.models import (
    ConstantSpectralModel,
    Models,
    GaussianSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    LightCurveTemplateTemporalModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data

LOCATION = EarthLocation(lon="-70d18m58.84s", lat="-24d41m0.34s", height="2000m")


@pytest.fixture()
def signal_model():
    spatial_model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
    )

    spectral_model = PowerLawSpectralModel(amplitude="1e-11 cm-2 s-1 TeV-1")

    t_max = 1000 * u.s

    time = np.arange(t_max.value) * u.s
    tau = u.Quantity("2e2 s")
    norm = np.exp(-time / tau)

    table = Table()
    table["TIME"] = time
    table["NORM"] = norm / norm.max()
    t_ref = Time("2000-01-01")
    table.meta = dict(MJDREFI=t_ref.mjd, MJDREFF=0, TIMEUNIT="s", TIMESYS="utc")
    temporal_model = LightCurveTemplateTemporalModel.from_table(table)

    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        temporal_model=temporal_model,
        name="test-source",
    )


@pytest.fixture()
def energy_dependent_temporal_sky_model():
    spatial_model = PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic")
    spectral_model = ConstantSpectralModel(const="1 cm-2 s-1 TeV-1")
    temporal_model = get_energy_dependent_temporal_model()
    model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
    )
    return model


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
def models_list():
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
def test_observations_events_sampler_time(
    tmpdir, observations, energy_dependent_temporal_sky_model
):
    models = Models(energy_dependent_temporal_sky_model)
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
    sampler.run(observations, models=models)


@requires_data()
def test_observations_events_sampler_parallel(tmpdir, observations, models_list):
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
    sampler.run(observations, models=models_list)

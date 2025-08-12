# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from gammapy.data import DataStore, Observation, ObservationsEventsSampler
from gammapy.data.pointing import FixedPointingInfo
from gammapy.datasets.tests.test_simulate import get_energy_dependent_temporal_model
from gammapy.irf import load_irf_dict_from_file
from gammapy.maps import MapAxis
from gammapy.modeling.models import (
    ConstantSpectralModel,
    FoVBackgroundModel,
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data

LOCATION = EarthLocation(lon="-70d18m58.84s", lat="-24d41m0.34s", height="2000m")


@pytest.fixture()
def models(signal_model):
    bkg_model = FoVBackgroundModel(dataset_name="test")
    return [signal_model, bkg_model]


@pytest.fixture()
@requires_data()
def energy_dependent_temporal_sky_model(models):
    models[0].spatial_model = PointSpatialModel(
        lon_0="0 deg", lat_0="0 deg", frame="galactic"
    )
    models[0].spectral_model = ConstantSpectralModel(const="1 cm-2 s-1 TeV-1")
    models[0].temporal_model = get_energy_dependent_temporal_model()
    return models[0]


@requires_data()
def test_observation_event_sampler(signal_model, tmp_path):
    from gammapy.datasets.simulate import ObservationEventSampler

    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs = datastore.get_observations()[0]

    # first test defaults with HESS
    # otherwise with CTA the EdispMap computation takes too much time and memory
    maker = ObservationEventSampler()

    sim_obs = maker.run(obs, None)
    assert sim_obs.events is not None
    assert len(sim_obs.events.table) > 0

    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
    )
    pointing = FixedPointingInfo(
        fixed_icrs=SkyCoord(83.63311446, 22.01448714, unit="deg", frame="icrs"),
    )
    time_start = Time("2021-11-20T03:00:00")
    time_stop = Time("2021-11-20T03:30:00")

    obs = Observation.create(
        pointing=pointing,
        location=LOCATION,
        obs_id=1,
        tstart=time_start,
        tstop=time_stop,
        irfs=irfs,
        deadtime_fraction=0.01,
    )

    dataset_kwargs = dict(
        spatial_width=5 * u.deg,
        spatial_bin_size=0.01 * u.deg,
        energy_axis=MapAxis.from_energy_bounds(
            10 * u.GeV, 100 * u.TeV, nbin=5, per_decade=True
        ),
        energy_axis_true=MapAxis.from_energy_bounds(
            10 * u.GeV, 100 * u.TeV, nbin=5, per_decade=True, name="energy_true"
        ),
    )
    maker = ObservationEventSampler(dataset_kwargs=dataset_kwargs)

    sim_obs = maker.run(obs, [signal_model])
    assert sim_obs.events is not None
    assert len(sim_obs.events.table) > 0


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

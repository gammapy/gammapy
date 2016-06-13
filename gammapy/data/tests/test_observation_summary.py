# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from ...data import DataStore, ObservationTableSummary, ObservationSummary
from ...data import ObservationStats, ObservationStatsList, ObservationList
from ...data import Target
from ...utils.testing import requires_data, requires_dependency
from ...extern.regions.shapes import CircleSkyRegion
from ...background import reflected_regions_background_estimate as refl
from ...image import ExclusionMask


@requires_data('gammapy-extra')
@pytest.fixture
def table_summary():
    data_store = DataStore.from_dir(
        '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    target_pos = SkyCoord(83.633083, 22.0145, unit='deg')
    return ObservationTableSummary(data_store.obs_table, target_pos)


@requires_data('gammapy-extra')
def test_str(table_summary):
    text = str(table_summary)
    assert 'Observation summary' in text


@requires_data('gammapy-extra')
def test_offset(table_summary):
    offset = table_summary.offset
    assert_allclose(offset.degree.mean(), 1., rtol=1.e-2)
    assert_allclose(offset.degree.std(), 0.5, rtol=1.e-2)


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_zenith(table_summary):
    table_summary.plot_zenith_distribution()


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_offset(table_summary):
    table_summary.plot_offset_distribution()


@requires_data('gammapy-extra')
@pytest.fixture
def obs_summary():
    datastore = DataStore.from_dir(
        '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    run_list = [23523, 23526, 23559, 23592]

    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)

    target = Target(position=pos, on_region=on_region,
                    name='Crab Nebula', tag='crab')

    mask = ExclusionMask.read(
        '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')

    obs_list = ObservationList([datastore.obs(_) for _ in run_list])
    obs_stats = ObservationStatsList()

    for index, run in enumerate(obs_list):
        bkg = refl(on_region, run.pointing_radec, mask, run.events)

        obs_stats.append(ObservationStats.from_target(run, target, bkg))

    summary = ObservationSummary(obs_stats)

    return summary


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_significance(obs_summary):
    obs_summary.plot_significance_vs_livetime()


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_excess(obs_summary):
    obs_summary.plot_excess_vs_livetime()


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_background(obs_summary):
    obs_summary.plot_background_vs_livetime()


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_gamma_rate(obs_summary):
    obs_summary.plot_gamma_rate()


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_background_rate(obs_summary):
    obs_summary.plot_background_rate()


@requires_data('gammapy-extra')
def test_obs_str(obs_summary):
    text = str(obs_summary)
    assert 'Observation summary' in text

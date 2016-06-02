# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.tests.helper import pytest
from numpy.testing import assert_allclose

from ...data import DataStore, ObservationTable, EventList, Target
from ...datasets import gammapy_extra
from ...image import ExclusionMask
from regions.shapes import CircleSkyRegion
from ...spectrum import SpectrumExtraction
from ...spectrum.spectrum_extraction import SpectrumObservationList, SpectrumObservation
from ...utils.energy import EnergyBounds
from ...utils.testing import requires_dependency, requires_data


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_extraction(tmpdir):
    # Construct w/o config file
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    on_region = CircleSkyRegion(center, radius)

    target = Target(center, on_region)
    target.obs_id = [23523, 23592]

    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)
    target.add_obs_from_store(ds)

    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = ExclusionMask.read(exclusion_file)

    target.estimate_background(method='reflected', exclusion = excl)

    bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')

    ana = SpectrumExtraction(target, e_reco=bounds)

    # test methods on SpectrumObservationList
    obslist = ana.observations
    assert len(obslist) == 2
    obs23523 = obslist.obs(23523)
    assert obs23523.on_vector.total_counts.value == 123
    new_list = [obslist.obs(_) for _ in [23523, 23592]]
    assert new_list[0].obs_id == 23523
    assert new_list[1].obs_id == 23592

@pytest.mark.xfail(reason='Should this still be supported?')
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_extraction_from_config(tmpdir):
    configfile = gammapy_extra.filename(
        'test_datasets/spectrum/spectrum_analysis_example.yaml')
    extraction = SpectrumExtraction.from_configfile(configfile)
    extraction.obs_table.remove_rows([2, 3])
    extraction.extract_spectrum()
    desired = extraction.observations
    configfile2 = str(tmpdir / 'config.yaml')
    extraction.write_configfile(configfile2)
    extraction2 = SpectrumExtraction.from_configfile(configfile2)
    extraction2.extract_spectrum()
    actual = extraction2.observations
    assert actual[0].on_vector.total_counts == desired[0].on_vector.total_counts

    #test total off list
    extraction.write_total_offlist()
    testlist = EventList.read(SpectrumExtraction.OFFLIST_FILE)
    assert len(testlist) == np.sum([o.off_vector.total_counts for o in desired])

@pytest.mark.xfail(reason='This needs regeneration of the test OGIP files')
@requires_data('gammapy-extra')
def test_observation_stacking():
    obs_table_file = gammapy_extra.filename(
        'datasets/hess-crab4_pha/observation_table.fits')

    obs_table = ObservationTable.read(obs_table_file)
    temp = SpectrumObservationList.from_observation_table(obs_table)

    observations = temp.get_obslist_from_ids([23523, 23592])
    spectrum_observation_grouped = SpectrumObservation.stack_observation_list(observations, 0)
    obs0 = observations[0]
    obs1 = observations[1]

    # Test sum on/off vector and alpha group
    sum_on_vector = obs0.on_vector.counts + obs1.on_vector.counts
    sum_off_vector = obs0.off_vector.counts + obs1.off_vector.counts
    alpha_times_off_tot = obs0.alpha * obs0.off_vector.total_counts + obs1.alpha * obs1.off_vector.total_counts
    total_off = obs0.off_vector.total_counts + obs1.off_vector.total_counts

    assert_allclose(spectrum_observation_grouped.on_vector.counts, sum_on_vector)
    assert_allclose(spectrum_observation_grouped.off_vector.counts, sum_off_vector)
    assert_allclose(spectrum_observation_grouped.alpha, alpha_times_off_tot / total_off)

    # Test arf group
    total_time = obs0.meta.livetime + obs1.meta.livetime
    arf_times_livetime = obs0.meta.livetime * obs0.effective_area.data \
                         + obs1.meta.livetime * obs1.effective_area.data
    assert_allclose(spectrum_observation_grouped.effective_area.data, arf_times_livetime / total_time)
    # Test rmf group
    rmf_times_arf_times_livetime = obs0.meta.livetime * obs0.effective_area.data \
                                   * obs0.energy_dispersion.pdf_matrix.T \
                                   + obs1.meta.livetime * obs1.effective_area.data \
                                     * obs1.energy_dispersion.pdf_matrix.T

    inan = np.isnan(rmf_times_arf_times_livetime / arf_times_livetime)
    pdf_expexted = rmf_times_arf_times_livetime / arf_times_livetime
    pdf_expexted[inan] = 0
    assert_allclose(spectrum_observation_grouped.energy_dispersion.pdf_matrix, pdf_expexted.T, atol=1e-6)

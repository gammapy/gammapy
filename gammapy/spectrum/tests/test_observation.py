# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ...data import ObservationTable
from ...spectrum import SpectrumObservation, SpectrumObservationList


@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
@requires_dependency('scipy')
def test_spectrum_observation():
    phafile = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23523.fits")
    obs = SpectrumObservation.read(phafile)
    obs.peek()

    energy_binning = obs.get_binning(4)
    assert_quantity_allclose(energy_binning[5], 879.954 * u.GeV, rtol=1e-3)


@pytest.mark.xfail(reason='This needs some changes to the API')
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

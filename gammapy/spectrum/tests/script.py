# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.tests.helper import pytest
from astropy.utils.compat import NUMPY_LT_1_9
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord, Angle
from gammapy.spectrum.results import SpectrumFitResult, SpectrumStats
from gammapy.utils.testing import requires_dependency, requires_data, SHERPA_LT_4_8
from gammapy.region import SkyCircleRegion
from gammapy.datasets import gammapy_extra
from gammapy.utils.scripts import read_yaml
from gammapy.utils.energy import EnergyBounds
from gammapy.image import ExclusionMask
from gammapy.data import DataStore
from gammapy.datasets import gammapy_extra
from gammapy.spectrum import (
    SpectrumExtraction,
    run_spectrum_extraction_using_config,
)
from gammapy.spectrum.spectrum_extraction import SpectrumObservationList
from gammapy.spectrum.spectrum_extraction import SpectrumObservation


def make_spectrum_extraction():
    # Construct w/o config file
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    on_region = SkyCircleRegion(pos=center, radius=radius)

    bkg_method = dict(type='reflected', n_min=2)

    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = ExclusionMask.from_fits(exclusion_file)

    bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')

    obs = [23523, 23559, 11111]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)

    ana = SpectrumExtraction(datastore=ds, obs_ids=obs, on_region=on_region,
                             bkg_method=bkg_method, exclusion=excl,
                             ebounds=bounds)
    return ana


def test_spectrum_extraction_grouping_from_an_observation_list():
    ana = make_spectrum_extraction()
    ana.extract_spectrum()
    spectrum_observation_grouped = SpectrumObservation.grouping_from_an_observation_list(ana.observations, 0)
    obs0 = ana.observations[0]
    obs1 = ana.observations[1]
    sum_on_vector = obs0.on_vector.counts + obs1.on_vector.counts
    sum_off_vector = obs0.off_vector.counts + obs1.off_vector.counts
    alpha_times_off_tot = obs0.alpha * obs0.off_vector.total_counts + obs1.alpha * obs1.off_vector.total_counts
    total_off = obs0.off_vector.total_counts+obs1.off_vector.total_counts
    total_time = obs0.meta.livetime + obs1.meta.livetime
    arf_times_livetime = obs0.meta.livetime * obs0.effective_area.effective_area \
                         + obs1.meta.livetime * obs1.effective_area.effective_area
    for i in range(40):
        rmf_times_arf_times_livetime=obs0.meta.livetime * obs0.effective_area.effective_area[i] \
                                 * obs0.energy_dispersion.pdf_matrix[i,:]   \
                         + obs1.meta.livetime * obs1.effective_area.effective_area[i]  \
                           * obs1.energy_dispersion.pdf_matrix[i,:]
        arf_times_livetime_test = obs0.meta.livetime * obs0.effective_area.effective_area[i] \
                         + obs1.meta.livetime * obs1.effective_area.effective_area[i]
        print(i)
        assert_allclose(spectrum_observation_grouped.energy_dispersion.pdf_matrix[i,:], rmf_times_arf_times_livetime / arf_times_livetime_test)
    assert_allclose(spectrum_observation_grouped.on_vector.counts, sum_on_vector)
    assert_allclose(spectrum_observation_grouped.off_vector.counts, sum_off_vector)
    assert_allclose(spectrum_observation_grouped.alpha, alpha_times_off_tot/ total_off)
    import IPython; IPython.embed()
    assert_allclose(spectrum_observation_grouped.effective_area.effective_area, arf_times_livetime / total_time)
    #assert_allclose(spectrum_observation_grouped.energy_dispersion.pdf_matrix[45,:], rmf_times_arf_times_livetime / arf_times_livetime_test)
    #assert_allclose(spectrum_observation_grouped.energy_dispersion.pdf_matrix, (rmf_times_arf_times_livetime / arf_times_livetime).T)


if __name__ == '__main__':
    test_spectrum_extraction_grouping_from_an_observation_list()

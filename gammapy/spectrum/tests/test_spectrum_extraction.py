# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.coordinates import SkyCoord, Angle

from ...data import DataStore
from ...datasets import gammapy_extra
from ...image import ExclusionMask
from ...region import SkyCircleRegion
from ...spectrum import SpectrumExtraction
from ...utils.energy import EnergyBounds
from ...utils.testing import requires_dependency, requires_data
from ...spectrum.spectrum_extraction import SpectrumObservationList

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

@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_extraction(tmpdir):
    # Construct w/o config file
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    on_region = SkyCircleRegion(pos=center, radius=radius)

    bkg_method = dict(type='reflected', n_min=2)

    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = ExclusionMask.from_fits(exclusion_file)

    bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')

    obs = [23523, 23559, 11111, 23592]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)

    ana = SpectrumExtraction(datastore=ds, obs_ids=obs, on_region=on_region,
                           bkg_method=bkg_method, exclusion=excl,
                           ebounds=bounds)

    #test methods on SpectrumObservationList
    obs = ana.observations
    assert len(obs) == 3
    obs23523 = obs.get_obslist_from_obsid([23523])[0]
    assert obs23523.on_vector.total_counts == 123
    new_list = obs.get_obslist_from_obsid([23523, 23592])
    assert new_list[0].obs_id == 23523
    assert new_list[1].obs_id == 23592


@requires_data('gammapy-extra')
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
    assert_allclose(spectrum_observation_grouped.on_vector.counts, sum_on_vector)
    assert_allclose(spectrum_observation_grouped.off_vector.counts, sum_off_vector)
    assert_allclose(spectrum_observation_grouped.alpha, alpha_times_off_tot/ total_off)


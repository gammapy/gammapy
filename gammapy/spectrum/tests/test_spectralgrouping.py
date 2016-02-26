# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.coordinates import SkyCoord, Angle
from numpy.testing import assert_allclose
from ...utils.energy import EnergyBounds
from ...image import ExclusionMask
from ...data import DataStore
from ...region import SkyCircleRegion
from ...datasets import gammapy_extra
from ...spectrum import SpectrumExtraction
from ...spectrum.spectral_grouping import SpectralGrouping
from ...spectrum.spectrum_fit import SpectrumFit
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_data

@requires_data('gammapy-extra')
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

    obs = [23523, 23559, 23526, 23592]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)

    ana = SpectrumExtraction(datastore=ds, obs_ids=obs, on_region=on_region,
                             bkg_method=bkg_method, exclusion=excl,
                             ebounds=bounds)
    return ana

def test_define_spectral_groups():
    ana= make_spectrum_extraction()
    group=SpectralGrouping(ana.observations)
    obs_groups = group.define_spectral_groups()
    assert obs_groups.n_groups == 25*30*40
    obs_groups2 =group.define_spectral_groups(offset_range=[0, 2.5], n_off_bin=1, eff_range=[0, 100], n_eff_bin=1, zen_range=[0., 70.], n_zen_bin=1)
    assert obs_groups2.n_groups == 1

 
def test_define_groups_and_stack(tmpdir):
    ana= make_spectrum_extraction()
    ana.observations.write_ogip_data(outdir=tmpdir+'ogip_data')
    obs_table = ana.observations.to_observation_table()

    fit = SpectrumFit.from_observation_table(obs_table)
    fit.model = 'PL'
    fit.energy_threshold_low = '100 GeV'
    fit.energy_threshold_high = '10 TeV'
    fit.run(method='sherpa')


    #Test that if we have 4 bands for the 4 runs it gives exactly the same result that before
    group=SpectralGrouping(ana.observations)
    obs_list =group.define_groups_and_stack()
    obs_list.write_ogip_data(outdir=tmpdir+'group_2')
    band_obs=obs_list.to_observation_table()
    fit_band2 = SpectrumFit.from_observation_table(band_obs)
    fit_band2.model = 'PL'
    fit_band2.energy_threshold_low = '100 GeV'
    fit_band2.energy_threshold_high = '10 TeV'
    fit_band2.run(method='sherpa')
    assert_quantity_allclose(fit.result.parameters["index"], fit_band2.result.parameters["index"])
    assert_quantity_allclose(fit.result.parameters["norm"], fit_band2.result.parameters["norm"])

    #Test that if we stack all the runs in one band we get a result close than before
    obs_list2 =group.define_groups_and_stack(offset_range=[0, 2.5], n_off_bin=1, eff_range=[0, 100], n_eff_bin=1,
                                             zen_range=[0., 70.], n_zen_bin=1)
    obs_list2.write_ogip_data(outdir=tmpdir+'group_all')
    band_obs2=obs_list2.to_observation_table()
    fit_band3 = SpectrumFit.from_observation_table(band_obs2)
    fit_band3.model = 'PL'
    fit_band3.energy_threshold_low = '100 GeV'
    fit_band3.energy_threshold_high = '10 TeV'
    fit_band3.run(method='sherpa')
    assert_quantity_allclose(fit.result.parameters["index"], fit_band3.result.parameters["index"], atol=1e-2)
    assert_allclose((fit_band3.result.parameters["norm"] - fit.result.parameters["norm"])
                             /fit.result.parameters["norm"], 0, atol=1e-2)




# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose, pytest
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from regions import CircleSkyRegion
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds
from ...data import Target, DataStore
from ...spectrum import SpectrumExtraction
from ...spectrum.models import PowerLaw
from ...background import ReflectedRegionsBackgroundEstimator
from ...image import SkyImage
from ..lightcurve import LightCurve, LightCurveEstimator


def test_lightcurve():
    lc = LightCurve.simulate_example()
    flux_mean = lc['FLUX'].mean()
    assert_quantity_allclose(flux_mean, Quantity(5.25, 'cm^-2 s^-1'))


def test_lightcurve_fvar():
    lc = LightCurve.simulate_example()
    fvar, fvar_err = lc.compute_fvar()
    assert_allclose(fvar, 0.6565905201197404)
    # Note: the following tolerance is very low in the next assert,
    # because results differ by ~ 1e-3 :
    # travis-ci result: 0.05773502691896258
    # Christoph's Macbook: 0.057795285237677206
    assert_allclose(fvar_err, 0.057795285237677206, rtol=1e-2)


@requires_dependency('matplotlib')
def test_lightcurve_plot():
    lc = LightCurve.simulate_example()
    lc.plot()


@requires_dependency('scipy')
def test_lightcurve_chisq():
    lc = LightCurve.simulate_example()
    chi2, pval = lc.compute_chisq()
    assert_quantity_allclose(chi2, 7.0)
    assert_quantity_allclose(pval, 0.07189777249646509)




# TODO: Reuse fixtures from spectrum tests
@pytest.fixture(scope='session')
def spec_extraction():
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    obs_ids = [23523, 23526]
    obs_list = data_store.obs_list(obs_ids)

    target_position = SkyCoord(ra=83.63308,
                               dec=22.01450,
                               unit='deg')
    on_region_radius = Angle('0.11 deg')
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)
    target = Target(on_region=on_region, name='Crab', tag='ana_crab')

    exclusion_file = '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits'
    allsky_mask = SkyImage.read(exclusion_file)
    exclusion_mask = allsky_mask.cutout(
        position=target.on_region.center,
        size=Angle('6 deg'),
    )
    bkg_estimator = ReflectedRegionsBackgroundEstimator(on_region=on_region,
                                                        obs_list=obs_list,
                                                        exclusion_mask=exclusion_mask)
    bkg_estimator.run()

    e_reco = EnergyBounds.equal_log_spacing(0.2, 100, 50, unit='TeV')  # fine binning
    e_true = EnergyBounds.equal_log_spacing(0.05, 100, 200, unit='TeV')
    extraction = SpectrumExtraction(obs_list=obs_list,
                                    bkg_estimate=bkg_estimator.result,
                                    containment_correction=False,
                                    e_reco=e_reco,
                                    e_true=e_true)
    extraction.run()
    extraction.define_energy_threshold('area_max', percent=10.0)
    return extraction


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_lightcurve_estimator():
    spec_extract = spec_extraction()
    lc_estimator = LightCurveEstimator(spec_extract)

    # param
    intervals = []
    for obs in spec_extract.obs_list:
        intervals.append([obs.events.time[0], obs.events.time[-1]])

    model = PowerLaw(
        index=2.3 * u.Unit(''),
        amplitude=3.4e-11 * u.Unit('1 / (cm2 s TeV)'),
        reference=1 * u.TeV,
    )

    lc = lc_estimator.light_curve(
        time_intervals=intervals,
        spectral_model=model,
        energy_range=[0.5, 100] * u.TeV,
    )

    assert_quantity_allclose(len(lc), 2)

    # TODO:
    # The uncommented values are with containment correction, this does not
    # work at the moment, try to reproduce them later
    #assert_allclose(lc['FLUX'][0].value, 5.70852574714e-11, rtol=1e-2)
    assert_allclose(lc['FLUX'][0].value, 2.8517243785145818e-11, rtol=1e-2)
    #assert_allclose(lc['FLUX'][-1].value, 6.16718031281e-11, rtol=1e-2)
    assert_allclose(lc['FLUX'][-1].value, 2.8626063613082577e-11, rtol=1e-2)

    assert_allclose(lc['FLUX_ERR'][0].value, 5.43450927144e-12, rtol=1e-2)
    #assert_allclose(lc['FLUX_ERR'][-1].value, 5.91581572415e-12, rtol=1e-2)
    assert_allclose(lc['FLUX_ERR'][-1].value, 5.288113122707022e-12, rtol=1e-2)

    # same but with threshold equal to 2 TeV
    lc = lc_estimator.light_curve(
        time_intervals=intervals,
        spectral_model=model,
        energy_range=[2, 100] * u.TeV,
    )

    #assert_allclose(lc['FLUX'][0].value, 1.02122885108e-11, rtol=1e-2)
    assert_allclose(lc['FLUX'][0].value, 1.826273620432445e-12, rtol=1e-2)

    # TODO: add test exercising e_reco selection
    # TODO: add asserts on all measured quantities

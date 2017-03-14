# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose, pytest
from ...utils.testing import requires_dependency, requires_data
from ..lightcurve import LightCurve, LightCurveFactory
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord, Angle
from ...spectrum import SpectrumExtraction
from ...spectrum.models import PowerLaw
from ...data import Target, DataStore
from ...image import SkyMask
from regions import CircleSkyRegion
from astropy.time import Time
import astropy.units as u

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


@requires_data('gammapy-extra')
@pytest.fixture(scope='session')
def spec_extraction():
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    obs_ids = [23523, 23526, 23559, 23592]
    obs_list = data_store.obs_list(obs_ids)
    
    target_position = SkyCoord(ra=83.63308,
                               dec=22.01450,
                               unit='deg',
                               frame='icrs')
    on_region_radius = Angle('0.11 deg')
    on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)
    target = Target(on_region=on_region, name='Crab', tag='ana_crab')

    exclusion_file = '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits'
    allsky_mask = SkyMask.read(exclusion_file)
    exclusion_mask = allsky_mask.cutout(
        position=target.on_region.center,
        size=Angle('6 deg'),
    )
    bkg_estimation = dict(
        method='reflected',
        exclusion=exclusion_mask,
    )
    
    extraction = SpectrumExtraction(target=target,
                                    obs=obs_list,
                                    background=bkg_estimation,
                                    containment_correction=True)
    extraction.estimate_background(extraction.background)
    extraction.define_energy_threshold('area_max', percent=10.0)
    return extraction

@requires_data('gammapy-extra')
def test_lightcurvefactory():

    lc_factory = LightCurveFactory(spec_extraction())

    # param
    t_start = Time(100.0, format='mjd')
    t_stop = Time(100000.0, format='mjd')
    energy_range = [0.5, 10] * u.TeV
    model = PowerLaw(index=2.3 * u.Unit(''),
                     amplitude=3.4e-11 * u.Unit('1 / (cm2 s TeV)'),
                     reference=1 * u.TeV)

    # build the light curve
    lc = lc_factory.light_curve(t_start=t_start,
                                t_stop=t_stop,
                                t_binning_type='obs',
                                spectral_model=model,
                                energy_range=energy_range)

    # ToDo, add real test when results are checked
    assert_quantity_allclose(len(lc), 4)

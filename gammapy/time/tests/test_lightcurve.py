# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
import astropy.units as u
from astropy.table import Table, Column
from astropy.tests.helper import assert_quantity_allclose
from regions import CircleSkyRegion
from ...utils.testing import requires_data, requires_dependency, mpl_savefig_check, assert_time_allclose
from ...utils.energy import EnergyBounds
from ...utils.time import time_ref_to_dict, time_relative_to_ref
from ...data import Target, DataStore
from ...spectrum import SpectrumExtraction
from ...spectrum.models import PowerLaw
from ...background import ReflectedRegionsBackgroundEstimator
from ...image import SkyImage
from ..lightcurve import LightCurve, LightCurveEstimator


# time time_min time_max flux flux_err flux_ul
# 48705.1757 48705.134 48705.2174 0.57 0.29 nan
# 48732.89195 48732.8503 48732.9336 0.39 0.29 nan
# 48734.0997 48734.058 48734.1414 0.48 0.29 nan
# 48738.98535 48738.9437 48739.027 nan nan 0.97
# 48741.0259 48740.9842 48741.0676 0.34 0.29 nan


@pytest.fixture(scope='session')
def lc():
    # table = Table()
    # time_ref = Time('2010-01-01')
    # meta = time_ref_to_dict(time_ref)

    meta = dict(TIMESYS='utc')

    table = Table(meta=meta, data=[
        Column(Time(['2010-01-02', '2010-01-05']).mjd, 'time'),
        Column(Time(['2010-01-01', '2010-01-03']).mjd, 'time_min'),
        Column(Time(['2010-01-03', '2010-01-10']).mjd, 'time_max'),
        Column([1e-11, 3e-11], 'flux', unit='cm-2 s-1'),
        Column([0.1e-11, 0.3e-11], 'flux_err', unit='cm-2 s-1'),
        Column([np.nan, 3.6e-11], 'flux_ul', unit='cm-2 s-1'),
    ])

    return LightCurve(table=table)


def test_lightcurve_repr(lc):
    assert repr(lc) == 'LightCurve(len=2)'


def test_lightcurve_properties_time(lc):
    assert lc.time_scale == 'utc'
    assert lc.time_format == 'mjd'

    # Time-related attributes
    time = lc.time
    assert time.scale == 'utc'
    assert time.format == 'mjd'
    assert_allclose(time.mjd, [55198, 55201])

    assert_allclose(lc.time_min.mjd, [55197, 55199])
    assert_allclose(lc.time_max.mjd, [55199, 55206])

    # Note: I'm not sure why the time delta has this scale and format
    time_delta = lc.time_delta
    assert time_delta.scale == 'tai'
    assert time_delta.format == 'jd'
    assert_allclose(time_delta.jd, [2, 7])


def test_lightcurve_properties_flux(lc):
    flux = lc.table['flux'].quantity
    assert flux.unit == 'cm-2 s-1'
    assert_allclose(flux.value, [1e-11, 3e-11])


# TODO: extend these tests to cover other time scales.
# In those cases, CSV should not round-trip because there
# is no header info in CSV to store the time scale!
@requires_dependency('yaml')
@pytest.mark.parametrize('format', ['fits', 'ascii.ecsv', 'ascii.csv'])
def test_lightcurve_read_write(tmpdir, lc, format):
    filename = str(tmpdir / 'spam')

    lc.write(filename, format=format)
    lc = LightCurve.read(filename, format=format)

    # Check if time-related info round-trips
    time = lc.time
    assert time.scale == 'utc'
    assert time.format == 'mjd'
    assert_allclose(time.mjd, [55198, 55201])


def test_lightcurve_fvar(lc):
    fvar, fvar_err = lc.compute_fvar()
    assert_allclose(fvar, 0.6982120021884471)
    # Note: the following tolerance is very low in the next assert,
    # because results differ by ~ 1e-3 between different machines
    assert_allclose(fvar_err, 0.07905694150420949, rtol=1e-2)


@requires_dependency('scipy')
def test_lightcurve_chisq(lc):
    chi2, pval = lc.compute_chisq()
    assert_quantity_allclose(chi2, 1.0000000000000001e-11)
    assert_quantity_allclose(pval, 0.999997476867478)


@requires_dependency('matplotlib')
def test_lightcurve_plot(lc):
    lc.plot()
    mpl_savefig_check()


# TODO: Reuse fixtures from spectrum tests
@pytest.fixture(scope='session')
def spec_extraction():
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    obs_ids = [23523, 23526]
    obs_list = data_store.obs_list(obs_ids)

    target_position = SkyCoord(ra=83.63308, dec=22.01450, unit='deg')
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
    extraction.compute_energy_threshold(method_lo='area_max', area_percent_lo=10.0)
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
    table = lc.table

    assert isinstance(lc.table['time_min'][0], type(intervals[0][0].value))

    assert_quantity_allclose(len(table), 2)

    # TODO:
    # The uncommented values are with containment correction, this does not
    # work at the moment, try to reproduce them later
    # assert_allclose(table['flux'][0], 5.70852574714e-11, rtol=1e-2)
    assert_allclose(table['flux'][0], 2.8517243785145818e-11, rtol=1e-2)
    # assert_allclose(table['flux'][-1], 6.16718031281e-11, rtol=1e-2)
    assert_allclose(table['flux'][-1], 2.8626063613082577e-11, rtol=1e-2)

    assert_allclose(table['flux_err'][0], 2.40524569797e-12, rtol=1e-2)
    # assert_allclose(table['flux_err'][-1], 5.91581572415e-12, rtol=1e-2)
    assert_allclose(table['flux_err'][-1], 2.3436731834e-12, rtol=1e-2)

    # same but with threshold equal to 2 TeV
    lc = lc_estimator.light_curve(
        time_intervals=intervals,
        spectral_model=model,
        energy_range=[2, 100] * u.TeV,
    )

    # assert_allclose(table['flux'][0], 1.02122885108e-11, rtol=1e-2)
    assert_allclose(table['flux'][0], 2.847377780063258e-11, rtol=1e-2)

    # TODO: add test exercising e_reco selection
    # TODO: add asserts on all measured quantities

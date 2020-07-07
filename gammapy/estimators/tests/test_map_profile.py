# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.datasets import MapDatasetOnOff
from gammapy.data import GTI
from gammapy.maps import MapAxis, WcsGeom
from gammapy.estimators import ExcessProfileEstimator
from gammapy.utils.regions import make_orthogonal_rectangle_sky_regions


def get_simple_dataset_on_off():
    axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=40, binsz=0.01, axes=[axis])
    dataset = MapDatasetOnOff.create(geom)
    dataset.mask_safe += 1
    dataset.counts += 5
    dataset.counts_off += 1
    dataset.acceptance += 1
    dataset.acceptance_off += 1
    dataset.exposure += 1000*u.m**2*u.s
    dataset.gti = GTI.create([0 * u.s], [5 * u.h], reference_time="2010-01-01T00:00:00")
    return dataset


def make_boxes(wcs):
    start_line = SkyCoord(0.08, -0.16, unit='deg', frame='icrs')
    end_line = SkyCoord(359.9, 0.16, unit='deg', frame='icrs')
    return make_orthogonal_rectangle_sky_regions(start_line, end_line, wcs, 0.1*u.deg, 8)


def make_horizontal_boxes(wcs):
    start_line = SkyCoord(0.08, 0.1, unit='deg', frame='icrs')
    end_line = SkyCoord(359.9, 0.1, unit='deg', frame='icrs')
    return make_orthogonal_rectangle_sky_regions(start_line, end_line, wcs, 0.1*u.deg, 8)


def test_profile_content():
    mapdataset_onoff = get_simple_dataset_on_off()
    wcs = mapdataset_onoff.counts.geom.wcs
    boxes, axis = make_horizontal_boxes(wcs)

    prof_maker = ExcessProfileEstimator(boxes, axis)
    imp_prof = prof_maker.run(mapdataset_onoff)

    assert_allclose(imp_prof.table[7]['x_min'], -0.0562, atol=1e-4)
    assert_allclose(imp_prof.table[7]['x_ref'], -0.0674, atol=1e-4)
    assert_allclose(imp_prof.table[7]['counts'], [100., 100.], atol=1e-2)
    assert_allclose(imp_prof.table[7]['excess'], [80., 80.], atol=1e-2)
    assert_allclose(imp_prof.table[7]['sqrt_ts'], [7.6302447, 7.6302447], atol=1e-5)
    assert_allclose(imp_prof.table[7]['errn'], [-10.747017, -10.747017], atol=1e-5)
    assert_allclose(imp_prof.table[0]['ul'], [115.171871, 115.171871], atol=1e-5)
    assert_allclose(imp_prof.table[0]['flux'], [7.99999987e-06, 8.00000010e-06], atol=1e-3)
    assert_allclose(imp_prof.table[0]['solid_angle'], [6.853891e-07, 6.853891e-07], atol=1e-5)

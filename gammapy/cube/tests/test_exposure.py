from __future__ import absolute_import, division, print_function, \
    unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...irf import EffectiveAreaTable2D
from ...datasets import gammapy_extra
from .. import exposure_cube
from .. import SkyCube


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_exposure_cube():
    aeff_filename = gammapy_extra.filename(
        'datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz')
    ccube_filename = gammapy_extra.filename(
        'datasets/hess-crab4-hd-hap-prod2/hess_events_simulated_023523_cntcube.fits')

    pointing = SkyCoord(83.633, 21.514, unit='deg')
    livetime = Quantity(1581.17, 's')
    aeff2d = EffectiveAreaTable2D.read(aeff_filename)
    count_cube = SkyCube.read(ccube_filename, format='fermi-counts')
    offset_max = Angle(2.2, 'deg')
    exp_cube = exposure_cube(pointing, livetime, aeff2d, count_cube,
                             offset_max=offset_max)
    exp_ref = Quantity(4.7e8, 'm^2 s')
    coordinates = exp_cube.sky_image_ref.coordinates()
    offset = coordinates.separation(pointing)

    assert np.shape(exp_cube.data)[1:] == np.shape(count_cube.data)[1:]
    assert np.shape(exp_cube.data)[0] == np.shape(count_cube.data)[0]
    assert exp_cube.wcs == count_cube.wcs
    assert_equal(count_cube.energy_axis.energy, exp_cube.energy_axis.energy)
    assert_quantity_allclose(np.nanmax(exp_cube.data), exp_ref, rtol=100)
    assert exp_cube.data.unit == exp_ref.unit
    assert exp_cube.data[:, offset > offset_max].sum() == 0

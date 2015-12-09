from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_dependency
from ...utils.testing import requires_data
from ...irf import exposure_cube
from ...data import SpectralCube
from ...irf import EffectiveAreaTable2D
from ...datasets import gammapy_extra


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_exposure_cube():
    exp_ref = Quantity(4.7e8, 'm^2 s')

    aeff_filename = gammapy_extra.filename("datasets/hess-crab4/hess_aeff_023523.fits.gz")
    ccube_filename = gammapy_extra.filename("datasets/hess-crab4/hess_events_simulated_023523_cntcube.fits")

    pointing = SkyCoord(83.633, 21.514, frame='fk5', unit='deg')
    livetime = Quantity(1581.17, 's')
    aeff2D = EffectiveAreaTable2D.read(aeff_filename)
    count_cube = SpectralCube.read_counts(ccube_filename)
    exp_cube = exposure_cube(pointing, livetime, aeff2D, count_cube)

    assert np.shape(exp_cube.data)[1:] == np.shape(count_cube.data)[1:]
    assert np.shape(exp_cube.data)[0] == np.shape(count_cube.data)[0] + 1
    assert exp_cube.wcs == count_cube.wcs
    assert_equal(count_cube.energy, exp_cube.energy)
    assert_allclose(exp_cube.data, exp_ref, rtol=100)
    assert exp_cube.data.unit == exp_ref.unit

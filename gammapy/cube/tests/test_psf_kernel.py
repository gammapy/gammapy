# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency
from ...irf import TablePSF
from ...maps import MapAxis, WcsGeom
from .. import PSFKernel


@requires_dependency("scipy")
def test_table_psf_to_kernel_map():
    sigma = 0.5 * u.deg
    binsz = 0.1 * u.deg
    geom = WcsGeom.create(binsz=binsz, npix=150)

    rad = Angle(np.linspace(0., 3 * sigma.to("deg").value, 100), "deg")
    table_psf = TablePSF.from_shape(shape="gauss", width=sigma, rad=rad)
    kernel = PSFKernel.from_table_psf(table_psf, geom)
    kernel_array = kernel.psf_kernel_map.data

    # Is normalization OK?
    assert_allclose(kernel_array.sum(), 1.0, atol=1e-5)

    # maximum at the center of map?
    ind = np.unravel_index(np.argmax(kernel_array, axis=None), kernel_array.shape)
    # absolute tolerance at 0.5 because of even number of pixel here
    assert_allclose(ind, geom.center_pix, atol=0.5)


@requires_dependency("scipy")
def test_psf_kernel_from_gauss_read_write(tmpdir):
    sigma = 0.5 * u.deg
    binsz = 0.1 * u.deg
    geom = WcsGeom.create(binsz=binsz, npix=150, axes=[MapAxis((0, 1, 2))])

    kernel = PSFKernel.from_gauss(geom, sigma)

    # Check that both maps are identical
    assert_allclose(kernel.psf_kernel_map.data[0], kernel.psf_kernel_map.data[1])

    # Is there an odd number of pixels
    assert_allclose(np.array(kernel.psf_kernel_map.geom.npix) % 2, 1)

    filename = str(tmpdir / "test_kernel.fits")
    # Test read and write
    kernel.write(filename, overwrite=True)
    newkernel = PSFKernel.read(filename)
    assert_allclose(kernel.psf_kernel_map.data, newkernel.psf_kernel_map.data)

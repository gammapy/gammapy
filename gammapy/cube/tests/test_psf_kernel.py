# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle
from ...maps import WcsNDMap, MapAxis, WcsGeom
from .. import PSFKernel, table_psf_to_kernel_map
from ...irf import TablePSF, EnergyDependentTablePSF


def test_table_psf_to_kernel_map():
    sigma = 0.5 * u.deg
    binsz = 0.1 * u.deg
    geom = WcsGeom.create(binsz=binsz, npix=150)

    rad = Angle(np.linspace(0., 3 * sigma, 100), 'deg')
    table_psf = TablePSF.from_shape(shape='gauss', width=sigma, rad=rad)
    kernel = table_psf_to_kernel_map(table_psf, geom)

    # Is normalization OK?
    assert_allclose(kernel.data.sum(), 1.0, atol=1e-5)

    # maximum at the center of map?
    ind = np.unravel_index(np.argmax(kernel.data, axis=None), kernel.data.shape)
    # absolute tolerance at 0.5 because of even number of pixel here
    assert_allclose(ind, geom.center_pix, atol=0.5)


def test_psf_kernel_from_gauss():
    sigma = 0.5 * u.deg
    binsz = 0.1 * u.deg
    geom = WcsGeom.create(binsz=binsz, npix=150, axes=[MapAxis((0, 1, 2))])

    kernel = PSFKernel.from_gauss(geom, sigma)

    # Check that both maps are identical
    assert_allclose(kernel.psf_kernel_map.data[0], kernel.psf_kernel_map.data[1])

    # Is there an odd number of pixels
    assert_allclose(np.array(kernel.psf_kernel_map.geom.npix) % 2, 1)


def test_psf_kernel_convolve():
    sigma = 0.5 * u.deg
    binsz = 0.05 * u.deg

    testmap = WcsNDMap.create(binsz=binsz, width=5 * u.deg)
    testmap.fill_by_coord(([1], [1]), weights=np.array([2]))

    kernel = PSFKernel.from_gauss(testmap.geom, sigma)

    conv_map = kernel.apply(testmap)

    # Is convolved map normalization OK
    assert_allclose(conv_map.data.sum(), 2.0, atol=1e-3)

    # Are the maxima at the same position?
    index_conv = np.unravel_index(np.argmax(conv_map.data, axis=None), conv_map.data.shape)
    index_test = np.unravel_index(np.argmax(testmap.data, axis=None), testmap.data.shape)
    assert_allclose(index_conv, index_test)

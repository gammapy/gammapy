# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ...irf import TablePSF, EnergyDependentMultiGaussPSF
from ...maps import Map, WcsNDMap, MapAxis, WcsGeom
from .. import PSFKernel


@requires_dependency('scipy')
def test_table_psf_to_kernel_map():
    sigma = 0.5 * u.deg
    binsz = 0.1 * u.deg
    geom = WcsGeom.create(binsz=binsz, npix=150)

    rad = Angle(np.linspace(0., 3 * sigma.to('deg').value, 100), 'deg')
    table_psf = TablePSF.from_shape(shape='gauss', width=sigma, rad=rad)
    kernel = PSFKernel.from_table_psf(table_psf, geom)
    kernel_array = kernel.psf_kernel_map.data

    # Is normalization OK?
    assert_allclose(kernel_array.sum(), 1.0, atol=1e-5)

    # maximum at the center of map?
    ind = np.unravel_index(np.argmax(kernel_array, axis=None), kernel_array.shape)
    # absolute tolerance at 0.5 because of even number of pixel here
    assert_allclose(ind, geom.center_pix, atol=0.5)


@requires_dependency('scipy')
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


@requires_dependency('scipy')
def test_psf_kernel_convolve():
    sigma = 0.5 * u.deg
    binsz = 0.05 * u.deg

    testmap = WcsNDMap.create(binsz=binsz, width=5 * u.deg)
    testmap.fill_by_coord(([1], [1]), weights=np.array([2]))

    kernel = PSFKernel.from_gauss(testmap.geom, sigma, max_radius=1.5 * u.deg)

    # is kernel size OK?
    assert kernel.psf_kernel_map.geom.npix[0] == 61
    # is kernel maximum at the center?
    assert kernel.psf_kernel_map.data[30, 30] == np.max(kernel.psf_kernel_map.data)

    conv_map = testmap.convolve(kernel)

    # Is convolved map normalization OK
    assert_allclose(conv_map.data.sum(), 2.0, atol=1e-3)

    # Is the maximum in the convolved map at the right position?
    assert conv_map.get_by_coord([1, 1]) == np.max(conv_map.data)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_energy_dependent_psf_kernel():
    # Define energy axis
    energy_axis = MapAxis.from_edges(np.logspace(-1., 1., 4), unit='TeV', name='energy')

    # Create WcsGeom and map
    geom = WcsGeom.create(binsz=0.02 * u.deg, width=4.0 * u.deg, axes=[energy_axis])
    some_map = Map.from_geom(geom)
    some_map.fill_by_coord([[0.2, 0.4], [-0.1, 0.6], [0.5, 3.6]])

    # TODO : build EnergyDependentTablePSF programmatically rather than using CTA 1DC IRF
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    table_psf = psf.to_energy_dependent_table_psf(theta=0.5 * u.deg)

    psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius=1 * u.deg)

    assert psf_kernel.psf_kernel_map.data.shape == (3, 101, 101)

    some_map_convolved = some_map.convolve(psf_kernel)

    assert_allclose(some_map_convolved.data.sum(axis=(1, 2)), np.array((0, 1, 1)))

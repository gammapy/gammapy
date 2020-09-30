# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.irf import PSFKernel, TablePSF
from gammapy.maps import MapAxis, WcsGeom


def test_table_psf_to_kernel_map():
    sigma = 0.5 * u.deg
    binsz = 0.1 * u.deg

    geom = WcsGeom.create(binsz=binsz, npix=150)

    rad = Angle(np.linspace(0.0, 3 * sigma.to("deg").value, 100), "deg")
    table_psf = TablePSF.from_shape(shape="gauss", width=sigma, rad=rad)
    kernel = PSFKernel.from_table_psf(table_psf, geom)

    kernel_array = kernel.psf_kernel_map.data

    # Is normalization OK?
    assert_allclose(kernel_array.sum(), 1.0, atol=1e-5)

    # maximum at the center of map?
    ind = np.unravel_index(np.argmax(kernel_array, axis=None), kernel_array.shape)
    # absolute tolerance at 0.5 because of even number of pixel here
    assert_allclose(ind, geom.center_pix, atol=0.5)


def test_psf_kernel_from_gauss_read_write(tmp_path):
    sigma = 0.5 * u.deg
    binsz = 0.1 * u.deg
    geom = WcsGeom.create(binsz=binsz, npix=150, axes=[MapAxis((0, 1, 2))])

    kernel = PSFKernel.from_gauss(geom, sigma)

    # Check that both maps are identical
    assert_allclose(kernel.psf_kernel_map.data[0], kernel.psf_kernel_map.data[1])

    # Is there an odd number of pixels
    assert_allclose(np.array(kernel.psf_kernel_map.geom.npix) % 2, 1)

    kernel.write(tmp_path / "tmp.fits", overwrite=True)
    kernel2 = PSFKernel.read(tmp_path / "tmp.fits")
    assert_allclose(kernel.psf_kernel_map.data, kernel2.psf_kernel_map.data)


def test_psf_kernel_to_image():
    sigma1 = 0.5 * u.deg
    sigma2 = 0.2 * u.deg
    binsz = 0.1 * u.deg

    axis = MapAxis.from_energy_bounds(1, 10, 2, unit="TeV", name="energy_true")
    geom = WcsGeom.create(binsz=binsz, npix=50, axes=[axis])

    rad = Angle(np.linspace(0.0, 1.5 * sigma1.to("deg").value, 100), "deg")
    table_psf1 = TablePSF.from_shape(shape="disk", width=sigma1, rad=rad)
    table_psf2 = TablePSF.from_shape(shape="disk", width=sigma2, rad=rad)

    kernel1 = PSFKernel.from_table_psf(table_psf1, geom)
    kernel2 = PSFKernel.from_table_psf(table_psf2, geom)

    kernel1.psf_kernel_map.data[1, :, :] = kernel2.psf_kernel_map.data[1, :, :]

    kernel_image_1 = kernel1.to_image()
    kernel_image_2 = kernel1.to_image(exposure=[1, 2])

    assert_allclose(kernel_image_1.psf_kernel_map.data.sum(), 1.0, atol=1e-5)
    assert_allclose(kernel_image_1.psf_kernel_map.data[0, 25, 25], 0.028415, atol=1e-5)
    assert_allclose(kernel_image_1.psf_kernel_map.data[0, 22, 22], 0.009806, atol=1e-5)
    assert_allclose(kernel_image_1.psf_kernel_map.data[0, 20, 20], 0.0, atol=1e-5)

    assert_allclose(kernel_image_2.psf_kernel_map.data.sum(), 1.0, atol=1e-5)
    assert_allclose(
        kernel_image_2.psf_kernel_map.data[0, 25, 25], 0.03791383, atol=1e-5
    )
    assert_allclose(kernel_image_2.psf_kernel_map.data[0, 22, 22], 0.0079069, atol=1e-5)
    assert_allclose(kernel_image_2.psf_kernel_map.data[0, 20, 20], 0.0, atol=1e-5)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.irf import PSFKernel
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import DiskSpatialModel


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

    disk_1 = DiskSpatialModel(r_0=sigma1)
    disk_2 = DiskSpatialModel(r_0=sigma2)

    rad_max = 2.5 * u.deg
    kernel1 = PSFKernel.from_spatial_model(disk_1, geom, max_radius=rad_max, factor=4)
    kernel2 = PSFKernel.from_spatial_model(disk_2, geom, max_radius=rad_max, factor=4)

    kernel1.psf_kernel_map.data[1, :, :] = kernel2.psf_kernel_map.data[1, :, :]

    kernel_image_1 = kernel1.to_image()
    kernel_image_2 = kernel1.to_image(exposure=[1, 2])

    assert_allclose(kernel_image_1.psf_kernel_map.data.sum(), 1.0, atol=1e-5)
    assert_allclose(kernel_image_1.psf_kernel_map.data[0, 25, 25], 0.02844, atol=1e-5)
    assert_allclose(kernel_image_1.psf_kernel_map.data[0, 22, 22], 0.009636, atol=1e-5)
    assert_allclose(kernel_image_1.psf_kernel_map.data[0, 20, 20], 0.0, atol=1e-5)

    assert_allclose(kernel_image_2.psf_kernel_map.data.sum(), 1.0, atol=1e-5)
    assert_allclose(
        kernel_image_2.psf_kernel_map.data[0, 25, 25], 0.038091, atol=1e-5
    )
    assert_allclose(kernel_image_2.psf_kernel_map.data[0, 22, 22], 0.00777, atol=1e-5)
    assert_allclose(kernel_image_2.psf_kernel_map.data[0, 20, 20], 0.0, atol=1e-5)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ...datasets import FermiVelaRegion
from .. import compute_npred_cube
from ..core import SkyCube
from .test_core import make_test_sky_cube


@requires_dependency('scipy')
@requires_dependency('reproject')
def test_analytical_npred_cube():
    sky_cube = make_test_sky_cube()

    # Choose exposure such that exposure * flux_int integrates to unity
    exposure_cube = SkyCube.empty(enumbins=4, nxpix=3, nypix=3, fill=1e12, unit='cm2 s')
    ebounds = [1, 100] * u.TeV

    solid_angle = exposure_cube.sky_image_ref.solid_angle()

    # Integral resolution is 1 as this is a true powerlaw case
    npred_cube = compute_npred_cube(sky_cube, exposure_cube, ebounds, integral_resolution=2)
    actual = npred_cube.data[0].value
    assert_quantity_allclose(actual, solid_angle.value)


@requires_dependency('scipy')
@requires_dependency('reproject')
@requires_data('gammapy-extra')
def test_compute_npred_cube():
    fermi_vela = FermiVelaRegion()

    background = fermi_vela.diffuse_model()
    exposure = fermi_vela.exposure_cube()

    # Re-project background cube
    repro_bg_cube = background.reproject(exposure)

    # Define energy band required for output
    ebounds = [10, 500] * u.GeV

    # Compute the predicted counts cube
    npred_cube = compute_npred_cube(repro_bg_cube, exposure, ebounds, integral_resolution=5)

    # Convolve with Energy-dependent Fermi LAT PSF
    psf = fermi_vela.psf()
    kernels = psf.kernels(npred_cube, rad_max=2 * u.deg)
    convolved_npred_cube = npred_cube.convolve(kernels)

    actual = convolved_npred_cube.data.value.sum()
    desired = fermi_vela.background_image().data.sum()

    assert_allclose(actual, desired, rtol=0.001)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing.utils import assert_allclose
from astropy.convolution import Tophat2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...detect import compute_lima_image, compute_lima_on_off_image
from ...maps import Map


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_compute_lima_image():
    """
    Test Li & Ma image against TS image for Tophat kernel
    """
    filename = (
        "$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz"
    )
    counts = Map.read(filename, hdu="counts")
    background = Map.read(filename, hdu="background")

    kernel = Tophat2DKernel(5)
    result_lima = compute_lima_image(counts, background, kernel)

    assert_allclose(result_lima["significance"].data[100, 100], 30.814916, atol=1e-3)
    assert_allclose(result_lima["significance"].data[1, 1], 0.164, atol=1e-3)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_compute_lima_on_off_image():
    """
    Test Li & Ma image with snippet from the H.E.S.S. survey data.
    """
    filename = (
        "$GAMMAPY_EXTRA/test_datasets/unbundled/hess/survey/hess_survey_snippet.fits.gz"
    )
    n_on = Map.read(filename, hdu="ON")
    n_off = Map.read(filename, hdu="OFF")
    a_on = Map.read(filename, hdu="ONEXPOSURE")
    a_off = Map.read(filename, hdu="OFFEXPOSURE")
    significance = Map.read(filename, hdu="SIGNIFICANCE")

    kernel = Tophat2DKernel(5)
    results = compute_lima_on_off_image(n_on, n_off, a_on, a_off, kernel)

    # Reproduce safe significance threshold from HESS software
    results["significance"].data[results["n_on"].data < 5] = 0

    # crop the image at the boundaries, because the reference image
    # is cut out from a large map, there is no way to reproduce the
    # result with regular boundary handling
    actual = results["significance"].crop(kernel.shape).data
    desired = significance.crop(kernel.shape).data

    # Set boundary to NaN in reference image
    assert_allclose(actual, desired, atol=1e-5)

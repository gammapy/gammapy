# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from ...utils.testing import requires_data
from ...maps import Map
from ..asmooth import ASmooth


@pytest.fixture(scope="session")
def input_maps():
    filename = "$GAMMAPY_DATA/tests/unbundled/poisson_stats_image/input_all.fits.gz"
    return {
        "counts": Map.read(filename, hdu="counts"),
        "background": Map.read(filename, hdu="background"),
    }


@requires_data()
def test_asmooth(input_maps):
    kernel = Tophat2DKernel
    scales = ASmooth.make_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmooth(kernel=kernel, scales=scales, method="simple", threshold=2.5)
    smoothed = asmooth.run(input_maps["counts"], input_maps["background"])

    desired = {
        "counts": 6.454327,
        "background": 1.0,
        "scale": 0.056419,
        "significance": 18.125747,
    }

    for name in smoothed:
        actual = smoothed[name].data[100, 100]
        assert_allclose(actual, desired[name], rtol=1e-5)

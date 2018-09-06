# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from ...utils.testing import requires_data, requires_dependency
from ...utils.scripts import make_path
from ...maps import Map
from ..asmooth import ASmooth


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_asmooth():
    filename = make_path("$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_vela.fits.gz")
    counts = Map.read(filename, hdu="Counts")
    background = Map.read(filename, hdu="BACKGROUND")

    kernel = Gaussian2DKernel
    scales = ASmooth.make_scales(15, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmooth(kernel=kernel, scales=scales[6:], method="lima", threshold=4)
    smoothed = asmooth.run(counts, background)

    desired = {
        "counts": 0.02089332998318483,
        "background": 0.022048139647973686,
        "scale": np.nan,
        "significance": np.nan,
    }

    for name in smoothed:
        actual = smoothed[name].data[100, 100]
        assert_allclose(actual, desired[name])

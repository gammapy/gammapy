# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from ...utils.testing import requires_data, requires_dependency
from .. import ASmooth, SkyImageList, asmooth_scales


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_asmooth():
    images = SkyImageList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_vela.fits.gz')
    images['Counts'].name = 'counts'
    images['BACKGROUND'].name = 'background'

    kernel = Gaussian2DKernel
    scales = asmooth_scales(15, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmooth(kernel=kernel, scales=scales[6:], method='lima', threshold=4)
    smoothed = asmooth.run(images)

    desired = {
        'counts': 0.02089332998318483,
        'background': 0.022048139647973686,
        'scale': np.nan,
        'significance': np.nan,
    }

    for name in smoothed.names:
        actual = smoothed[name].data[100, 100]
        assert_allclose(actual, desired[name])

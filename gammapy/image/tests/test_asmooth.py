# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from .. import SkyImage, ASmooth, SkyImageList, asmooth_scales
import astropy.units as u
from astropy.convolution import Gaussian2DKernel



def test_asmooth():
	images = SkyImageList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_vela.fits.gz')
	images['COUNTS'].name = 'counts'
	images['BACKGROUND'].name = 'background'

	kernel = Gaussian2DKernel
	scales = asmooth_scales(15, kernel=kernel) * 0.1 * u.deg

	asmooth = ASmooth(kernel=kernel, scales=scales[6:], method='lima', threshold=4)

	smoothed = asmooth.run(images)
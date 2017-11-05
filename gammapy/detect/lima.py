# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy
import logging
import numpy as np
from ..image import SkyImage, SkyImageList
from ..stats import significance, significance_on_off

__all__ = [
    'compute_lima_image',
    'compute_lima_on_off_image',
]

log = logging.getLogger(__name__)


def compute_lima_image(counts, background, kernel, exposure=None):
    """Compute Li & Ma significance and flux images for known background.

    If exposure is given the corresponding flux image is computed and returned.

    Parameters
    ----------
    counts : `~gammapy.image.SkyImage`
        Counts image
    background : `~gammapy.image.SkyImage`
        Background image
    kernel : `astropy.convolution.Kernel2D`
        Convolution kernel
    exposure : `~gammapy.image.SkyImage`
        Exposure image

    Returns
    -------
    images : `~gammapy.image.SkyImageList`
        Results images container

    See Also
    --------
    gammapy.stats.significance
    """
    from scipy.ndimage import convolve

    wcs = counts.wcs.copy()
    # Kernel is modified later make a copy here
    kernel = deepcopy(kernel)

    kernel.normalize('peak')
    conv_opt = dict(mode='constant', cval=np.nan)

    counts_conv = convolve(counts, kernel.array, **conv_opt)
    background_conv = convolve(background, kernel.array, **conv_opt)
    excess_conv = counts_conv - background_conv
    significance_conv = significance(counts_conv, background_conv, method='lima')

    images = SkyImageList([
        SkyImage(name='significance', data=significance_conv, wcs=wcs),
        SkyImage(name='counts', data=counts_conv, wcs=wcs),
        SkyImage(name='background', data=background_conv, wcs=wcs),
        SkyImage(name='excess', data=excess_conv, wcs=wcs),
    ])

    # TODO: should we be doing this here?
    # Wouldn't it be better to let users decide if they want this,
    # and have it easily accessible as an attribute or method?
    _add_other_images(images, exposure, kernel, conv_opt)

    return images


def compute_lima_on_off_image(n_on, n_off, a_on, a_off, kernel, exposure=None):
    """Compute Li & Ma significance and flux images for on-off observations.

    Parameters
    ----------
    n_on : `~gammapy.image.SkyImage`
        Counts image
    n_off : `~gammapy.image.SkyImage`
        Off counts image
    a_on : `~gammapy.image.SkyImage`
        Relative background efficiency in the on region
    a_off : `~gammapy.image.SkyImage`
        Relative background efficiency in the off region
    kernel : `astropy.convolution.Kernel2D`
        Convolution kernel
    exposure : `~gammapy.image.SkyImage`
        Exposure image

    Returns
    -------
    images : `~gammapy.image.SkyImageList`
        Results images container

    See also
    --------
    gammapy.stats.significance_on_off
    """
    from scipy.ndimage import convolve

    # Kernel is modified later make a copy here
    kernel = deepcopy(kernel)

    kernel.normalize('peak')
    conv_opt = dict(mode='constant', cval=np.nan)

    n_on_conv = convolve(n_on.data, kernel.array, **conv_opt)
    a_on_conv = convolve(a_on.data, kernel.array, **conv_opt)
    alpha_conv = a_on_conv / a_off
    background_conv = alpha_conv * n_off.data
    excess_conv = n_on_conv - background_conv
    significance_conv = significance_on_off(n_on_conv, n_off.data, alpha_conv, method='lima')

    images = SkyImageList([
        SkyImage(name='significance', data=significance_conv),
        SkyImage(name='n_on', data=n_on_conv),
        SkyImage(name='background', data=background_conv),
        SkyImage(name='excess', data=excess_conv),
        SkyImage(name='alpha', data=alpha_conv),
    ])

    # TODO: should we be doing this here?
    # Wouldn't it be better to let users decide if they want this,
    # and have it easily accessible as an attribute or method?
    _add_other_images(images, exposure, kernel, conv_opt)

    return images


def _add_other_images(images, exposure, kernel, conv_opt):
    if not exposure:
        return

    from scipy.ndimage import convolve
    kernel.normalize('integral')
    exposure_conv = convolve(exposure, kernel.array, **conv_opt)
    flux = images['excess'].data / exposure_conv
    images['flux'] = SkyImage(name='flux', data=flux, wcs=images['excess'].wcs)

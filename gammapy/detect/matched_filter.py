# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Matched filter source detection methods"""
from __future__ import print_function, division
import numpy as np
from ..image import process_image_pixels
from ..stats import p_to_s

__all__ = ['probability_center',
           'probability_image',
           'significance_center',
           'significance_image',
           ]


def probability_center(images, kernel):
    """Compute matched-filter p-value at the kernel center.

    The shapes of the images and the kernel must match.

    Reference: Appendix B.1.2 from Stewart (2009)
        http://adsabs.harvard.edu/abs/2009A%26A...495..989S

    Parameters
    ----------
    images : dict of arrays
        Keys: 'counts', 'background'
    kernel : array_like
        Kernel array

    Returns
    -------
    probability : float
        Probability that counts is not a background fluctuation
    """
    from scipy.special import gammaincc as Q

    C = np.asanyarray(images['counts'])
    B = np.asanyarray(images['background'])
    w = np.asanyarray(kernel)

    assert C.shape == w.shape
    assert B.shape == w.shape

    # Normalize kernel
    w = w / w.sum()

    U = np.sum(w * C)
    B_prime = np.sum(w * B)
    w_equiv = np.sum(w * w * B) / B_prime
    P = Q(B_prime / w_equiv, U / w_equiv)
    return P


def significance_center(images, kernel):
    """Compute matched-filter significance at the kernel center.

    See `probability_center` docstring.
    """
    probability = probability_center(images, kernel)
    return p_to_s(probability)


def probability_image(images, kernel):
    """Compute matched-filter p-value image.

    Parameters
    ----------
    images : dict of arrays
        Keys: 'counts', 'background'

    kernel : array_like

    Returns
    -------
    probability : array
    """
    out = np.zeros_like(images['counts'], dtype='float64')
    process_image_pixels(images, kernel, out, probability_center)
    return out


def significance_image(images, kernel):
    """Compute matched-filter significance image.

    See `probability_image` docstring.
    """
    probability = probability_image(images, kernel)
    significance = p_to_s(probability)
    return significance

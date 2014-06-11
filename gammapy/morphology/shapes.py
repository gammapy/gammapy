# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Morphological models for astrophysical sources.

This was written before I used sherpa and is independent.
Might be useful to keep around anyways.
"""
from __future__ import print_function, division
import numpy as np
from numpy import sqrt, exp, sin, cos

__all__ = ['delta2d', 'gauss2d', 'shell2d', 'sphere2d',
           'morph_types', 'morph_pars']


def _normalized(image, ampl):
    """Normalize image such that image.sum() == ampl.

    If for the input image.sum() == 0, then do nothing.
    """
    sum = image.sum()
    if sum == 0:
        return image
    else:
        return (ampl / sum) * image


def delta2d(pars, x, y):
    """Point source."""
    xpos, ypos, ampl = pars

    # FIXME: This implementation of a delta function is quite inelegant.
    # Maybe there's a better and faster solution.
    r = sqrt((x - xpos) ** 2 + (y - ypos) ** 2)
    im = np.zeros_like(x)
    im[np.unravel_index(np.argmin(r), r.shape)] = ampl

    return _normalized(im, ampl)


def gauss2d(pars, x, y):
    """Asymmetric Gaussian."""
    xpos, ypos, ampl, sigma, epsilon, theta = pars

    # FIXME: hack to avoid division by 0
    # Should we do an if() and call delta2d()?
    sigma += 1e-3

    x_new = (x - xpos) * cos(theta) + (y - ypos) * sin(theta)
    y_new = (y - ypos) * cos(theta) + (x - xpos) * sin(theta)
    r = sqrt(x_new ** 2 * (1 - epsilon) ** 2 + (y_new ** 2)) / (1 - epsilon)
    im = exp(-0.5 * (r / sigma) ** 2)

    return _normalized(im, ampl)


def shell2d(pars, x, y):
    """Homogeneous radiating shell.

    Can be used as a toy shell-type SNR model.
    """
    xpos, ypos, ampl, r_out, r_in = pars

    r = sqrt((x - xpos) ** 2 + (y - ypos) ** 2)
    # Note: for r > r_out 'np.select' fills automatically zeros!
    # We only call abs() in sqrt() to avoid warning messages.
    im = np.select([r <= r_in, r <= r_out],
                   [sqrt(abs(r_out ** 2 - r ** 2)) - 
                    sqrt(abs(r_in ** 2 - r ** 2)),
                    sqrt(abs(r_out ** 2 - r ** 2))])

    # Compute integral of x distribution;
    # we then divide by this integral to get a properly normalized
    # surface brightness.
    # I did this with Mathematica and there is a test in place
    # that proves that this is the correct normalization.
    # integral = 2 * pi / 3 * (r_out ** 3 - r_in ** 3)

    # FIXME: this will have significant error from binning if the
    # source is only a few pixels large!
    # return ampl * im * (x[0, 0] - x[0, 1]) ** 2 / integral
    return _normalized(im, ampl)


def sphere2d(pars, x, y):
    """Homogeneous radiating sphere.

    Can be used as a toy PWN model.
    """
    xpos, ypos, ampl, r_out = pars

    r = sqrt((x - xpos) ** 2 + (y - ypos) ** 2)
    im = np.select([r < r_out, r >= r_out], [sqrt(r_out ** 2 - r ** 2), 0])
    return _normalized(im, ampl)

# Morphology parameter names
delta2d_par = ['xpos', 'ypos', 'ampl']
gauss2d_par = ['xpos', 'ypos', 'ampl', 'sigma', 'epsilon', 'theta']
shell2d_par = ['xpos', 'ypos', 'ampl', 'r_in', 'r_out']
sphere2d_par = ['xpos', 'ypos', 'ampl', 'r_out']

# Available morphology types
morph_types = {'delta2d': delta2d,
               'gauss2d': gauss2d,
               'shell2d': shell2d,
               'sphere2d': sphere2d}

# Union of all morphology parameters
morph_pars = set(delta2d_par + gauss2d_par +
                 shell2d_par + sphere2d_par)

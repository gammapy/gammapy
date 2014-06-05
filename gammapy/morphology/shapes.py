# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Morphological models for astrophysical sources.

This was written before I used sherpa and is independent.
Might be useful to keep around anyways.
"""
from __future__ import print_function, division
import numpy as np
from numpy import sqrt, exp, sin, cos

from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.models import Gaussian2D 

__all__ = ['delta2d', 'gauss2d', 'shell2d', 'sphere2d',
           'morph_types', 'morph_pars', 'Shell2D', 'Sphere2D',
           'Delta2D', 'Gaussian2D']


class Shell2D(Fittable2DModel):
    """
    Projected homogeneous radiating shell model.

    This model can be used for a shell type SNR source morphology.

    Parameters
    ----------
    amplitude : float
        Peak value of the shell function
    x_0 : float
        x position center of the shell
    y_0 : float
        y position center of the shell
    R_in : float
        Inner radius of the shell
    R_out : float
        Outer radius of the shell

    See Also
    --------
    Sphere2D, Delta2D, Gaussian2D

    Notes
    -----
    Model formula:

    .. math::

        f(r) = A \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{R_{out}^2 - r^2} - \\sqrt{R_{in}^2 - r^2} & : r < R_{in} \\\\
                    \\sqrt{R_{out}^2 - r^2} & :  R_{in} \\leq r \\leq R_{out} \\\\
                    0 & : r > R_{out}
                \\end{array}
            \\right.
    """

    amplitude = Parameter()
    x_0 = Parameter()
    y_0 = Parameter()
    R_in = Parameter()
    R_out = Parameter()

    def __init__(self, amplitude, x_0, y_0, R_in, R_out, **constraints):
        super(Shell2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                     y_0=y_0, R_in=R_in, R_out=R_out, **constraints)

    @staticmethod
    def eval(x, y, amplitude, x_0, y_0, R_in, R_out):
        """Two dimensional Shell model function"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        RR_in = R_in ** 2
        RR_out = R_out ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        np.seterr(invalid='ignore')

        # Note: for r > r_out 'np.select' fills automatically zeros!
        values = np.select([rr <= RR_in, rr <= RR_out],
                       [sqrt(RR_out - rr) - sqrt(RR_in - rr),
                        sqrt(RR_out - rr)])
        return amplitude * values


class Sphere2D(Fittable2DModel):
    """
    Projected homogeneous radiating sphere model.

    This model can be used for a simple PWN source morphology.

    Parameters
    ----------
    amplitude : float
        Peak value of the sphere function
    x_0 : float
        x position center of the sphere
    y_0 : float
        y position center of the sphere
    R_0 : float
        Inner radius of the sphere

    See Also
    --------
    Shell2D, Delta2D, Gaussian2D

    Notes
    -----
    Model formula:

    .. math::

        f(r) = A \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{R_0^2 - r^2} & :  r \\leq R_0 \\\\
                    0 & : r > R_0
                \\end{array}
            \\right.
    """

    amplitude = Parameter()
    x_0 = Parameter()
    y_0 = Parameter()
    R_0 = Parameter()

    def __init__(self, amplitude, x_0, y_0, R_0, **constraints):
        super(Sphere2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                     y_0=y_0, R_0=R_0, **constraints)

    @staticmethod
    def eval(x, y, amplitude, x_0, y_0, R_0):
        """Two dimensional Sphere model function"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        RR_0 = R_0 ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        np.seterr(invalid='ignore')

        values = np.select([rr <= RR_0, rr > RR_0], [2 * sqrt(RR_0 - rr), 0])
        return amplitude * values


class Delta2D(Fittable2DModel):
    """
    Two dimensional delta function .

    This model can be used for a point source morphology.

    Parameters
    ----------
    amplitude : float
        Peak value of the sphere function
    x_0 : float
        x position center of the sphere
    y_0 : float
        y position center of the sphere

    See Also
    --------
    Shell2D, Sphere2D, Gaussian2D

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = \\cdot \\left \\{
                    \\begin{array}{ll}
                        A & :  x = x_0 \\ \\mathrm{and} \\ y = y_0 \\\\
                        0 & : else
                    \\end{array}
                \\right.

    The pixel positions x_0 and y_0 are rounded to integers. Subpixel
    information is lost.
    """

    amplitude = Parameter()
    x_0 = Parameter()
    y_0 = Parameter()

    def __init__(self, amplitude, x_0, y_0, **constraints):
        super(Delta2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                     y_0=y_0, **constraints)

    @staticmethod
    def eval(x, y, amplitude, x_0, y_0):
        """Two dimensional delta model function"""
        dx = x - x_0
        dy = y - y_0
        x_mask = np.logical_and(dx > -0.5, dx <= 0.5)
        y_mask = np.logical_and(dy > -0.5, dy <= 0.5)
        return np.select([np.logical_and(x_mask, y_mask)], [amplitude])



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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Morphological models for astrophysical sources.

This was written before I used sherpa and is independent.
Might be useful to keep around anyways.
"""
from __future__ import print_function, division
import numpy as np
from numpy import sqrt, exp, sin, cos

from astropy.modeling import Fittable2DModel, Parameter, ModelDefinitionError
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
        Value of the integral of the shell function.
    x_0 : float
        x position center of the shell
    y_0 : float
        y position center of the shell
    r_in : float
        Inner radius of the shell
    width : float
        Width of the shell
    r_out : float (optional)
        Outer radius of the shell
    normed : bool (True)
        If set the amplitude parameter  corresponds to the integral of the
        function. If not set the 'amplitude' parameter corresponds to the
        peak value of the function (value at :math:`r = r_{in}`).

    See Also
    --------
    Sphere2D, Delta2D, Gaussian2D

    Notes
    -----
    Model formula with integral normalization:

    .. math::

        f(r) = A \\frac{3}{2 \\pi (r_{out}^3 - r_{in}^3)} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_{out}^2 - r^2} - \\sqrt{r_{in}^2 - r^2} & : r < r_{in} \\\\
                    \\sqrt{r_{out}^2 - r^2} & :  r_{in} \\leq r \\leq r_{out} \\\\
                    0 & : r > r_{out}
                \\end{array}
            \\right.

    Model formula with peak normalization:

    .. math::

        f(r) = A \\frac{1}{\\sqrt{r_{out}^2 - r_{in}^2}} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_{out}^2 - r^2} - \\sqrt{r_{in}^2 - r^2} & : r < r_{in} \\\\
                    \\sqrt{r_{out}^2 - r^2} & :  r_{in} \\leq r \\leq r_{out} \\\\
                    0 & : r > r_{out}
                \\end{array}
            \\right.

    With :math:`r_{out} = r_{in} + \\mathrm{width}`.
    """

    amplitude = Parameter()
    x_0 = Parameter()
    y_0 = Parameter()
    r_in = Parameter()
    width = Parameter()

    def __init__(self, amplitude, x_0, y_0, r_in, width=None, r_out=None, normed=True, **constraints):
        if r_out is not None:
            width = r_out - r_in
        if r_out is None and width is None:
            raise ModelDefinitionError("Either specify width or r_out.")

        if not normed:
            self.eval = self.eval_peak_norm
        super(Shell2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                     y_0=y_0, r_in=r_in, width=width, **constraints)

    @staticmethod
    def eval(x, y, amplitude, x_0, y_0, r_in, width):
        """Two dimensional Shell model function normed to integral"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_in = r_in ** 2
        rr_out = (r_in + width) ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        np.seterr(invalid='ignore')

        # Note: for r > r_out 'np.select' fills automatically zeros!
        values = np.select([rr <= rr_in, rr <= rr_out],
                       [sqrt(rr_out - rr) - sqrt(rr_in - rr),
                        sqrt(rr_out - rr)])
        return amplitude * values / (2 * np.pi / 3 *
                                     (rr_out * (r_in + width) - rr_in * r_in))

    @staticmethod
    def eval_peak_norm(x, y, amplitude, x_0, y_0, r_in, width):
        """Two dimensional Shell model function normed to peak value"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_in = r_in ** 2
        rr_out = (r_in + width) ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        np.seterr(invalid='ignore')

        # Note: for r > r_out 'np.select' fills automatically zeros!
        values = np.select([rr <= rr_in, rr <= rr_out],
                       [sqrt(rr_out - rr) - sqrt(rr_in - rr),
                        sqrt(rr_out - rr)])
        return amplitude * values / np.sqrt(rr_out - rr_in)


class Sphere2D(Fittable2DModel):
    """
    Projected homogeneous radiating sphere model.

    This model can be used for a simple PWN source morphology.

    Parameters
    ----------
    amplitude : float
        Value of the integral of the sphere function
    x_0 : float
        x position center of the sphere
    y_0 : float
        y position center of the sphere
    r_0 : float
        Radius of the sphere
    normed : bool (True)
        If set the amplitude parameter corresponds to the integral of the
        function. If not set the 'amplitude' parameter corresponds to the
        peak value of the function (value at :math:`r = 0`).


    See Also
    --------
    Shell2D, Delta2D, Gaussian2D

    Notes
    -----
    Model formula with integral normalization:

    .. math::

        f(r) = A \\frac{3}{4 \\pi r_0^3} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_0^2 - r^2} & :  r \\leq r_0 \\\\
                    0 & : r > r_0
                \\end{array}
            \\right.

    Model formula with peak normalization:

    .. math::

        f(r) = A \\frac{1}{r_0} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_0^2 - r^2} & :  r \\leq r_0 \\\\
                    0 & : r > r_0
                \\end{array}
            \\right.
    """

    amplitude = Parameter()
    x_0 = Parameter()
    y_0 = Parameter()
    r_0 = Parameter()

    def __init__(self, amplitude, x_0, y_0, r_0, normed=True, **constraints):
        if not normed:
            self.eval = self.eval_peak_norm
        super(Sphere2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                     y_0=y_0, r_0=r_0, **constraints)

    @staticmethod
    def eval(x, y, amplitude, x_0, y_0, r_0):
        """Two dimensional Sphere model function normed to integral"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_0 = r_0 ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        np.seterr(invalid='ignore')

        values = np.select([rr <= rr_0, rr > rr_0], [2 * sqrt(rr_0 - rr), 0])
        return amplitude * values / (4 / 3. * np.pi * rr_0 * r_0)

    @staticmethod
    def eval_peak_norm(x, y, amplitude, x_0, y_0, r_0):
        """Two dimensional Sphere model function normed to peak value"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_0 = r_0 ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        np.seterr(invalid='ignore')

        values = np.select([rr <= rr_0, rr > rr_0], [sqrt(rr_0 - rr), 0])
        return amplitude * values / r_0


class Delta2D(Fittable2DModel):
    """
    Two dimensional delta function .

    This model can be used for a point source morphology.

    Parameters
    ----------
    amplitude : float
        Peak value of the point source
    x_0 : float
        x position center of the point source
    y_0 : float
        y position center of the point source

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

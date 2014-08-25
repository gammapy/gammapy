# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Morphological models for astrophysical sources.

This was written before I used sherpa and is independent.
Might be useful to keep around anyways.
"""
from __future__ import print_function, division
import numpy as np
from astropy.modeling import (Parameter,
                              ModelDefinitionError,
                              Fittable2DModel,
                              )
from astropy.modeling.models import Gaussian2D

__all__ = ['morph_types', 'Shell2D', 'Sphere2D', 'Delta2D']


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
    Sphere2D, Delta2D, astropy.modeling.models.Gaussian2D

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

    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')
    r_in = Parameter('r_in')
    width = Parameter('width')

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
        # Note: for r > r_out 'np.select' fills automatically zeros!
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_in, rr <= rr_out],
                           [np.sqrt(rr_out - rr) - np.sqrt(rr_in - rr),
                            np.sqrt(rr_out - rr)])
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
        # Note: for r > r_out 'np.select' fills automatically zeros!
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_in, rr <= rr_out],
                           [np.sqrt(rr_out - rr) - np.sqrt(rr_in - rr),
                            np.sqrt(rr_out - rr)])
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
    Shell2D, Delta2D, astropy.modeling.models.Gaussian2D

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

    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')
    r_0 = Parameter('r_0')

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
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_0, rr > rr_0], [2 * np.sqrt(rr_0 - rr), 0])
        return amplitude * values / (4 / 3. * np.pi * rr_0 * r_0)

    @staticmethod
    def eval_peak_norm(x, y, amplitude, x_0, y_0, r_0):
        """Two dimensional Sphere model function normed to peak value"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_0 = r_0 ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_0, rr > rr_0], [np.sqrt(rr_0 - rr), 0])
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
    Shell2D, Sphere2D, astropy.modeling.models.Gaussian2D

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

    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')

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

# Available morphology types
morph_types = {'delta2d': Delta2D,
               'gauss2d': Gaussian2D,
               'shell2d': Shell2D,
               'sphere2d': Sphere2D}

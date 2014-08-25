# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Classes for working with radial distributions,
e.g. the PSF or a source or a PSF-convolved source.

@todo: ThetaCalculator2D and ModelThetaCalculator are not
finished and need tests!
"""
from __future__ import print_function, division
import numpy as np

__all__ = ['ModelThetaCalculator',
           'ThetaCalculator',
           'ThetaCalculator2D',
           'ThetaCalculatorScipy',
           ]


class ThetaCalculator(object):
    """Provides methods ``containment_fraction(theta)`` and ``containment_radius(containment_fraction)``
    given some 1D distribution (not necessarily normalized).

    Notes
    If you have to compute theta or containment many times for
    the same dist, this is much faster than ThetaCalculatorScipy.
    If you want only one value it could actually be slower,
    especially the containment calculation.

    Parameters
    ----------
    dist : callable
        Distribution function dp / dtheta2 (theta2)
    theta_max : float
        Integration range will be 0 .. theta_max ^ 2
    nbins : int
        Integration step size
    normalize : bool
        Normalize discretized distribution to 1?
    """
    def __init__(self, dist, theta_max, n_bins, normalize=False):
        theta2 = np.linspace(0, theta_max ** 2, n_bins)
        dtheta2 = theta2[1] - theta2[0]
        p = (dist(theta2) * dtheta2).cumsum()
        if normalize:
            p /= p[-1]
        self.theta2, self.p = theta2, p

    def containment_fraction(self, theta):
        """Compute containment fraction for a given theta."""
        theta = np.asarray(theta)
        index = np.where(self.theta2 > theta ** 2)[0][0]
        return self.p[index]

    def containment_radius(self, containment_fraction):
        """Compute theta for a given containment fraction."""
        containment_fraction = np.asarray(containment_fraction)
        index = np.where(self.p > containment_fraction)[0][0]
        return np.sqrt(self.theta2[index])


class ThetaCalculatorScipy(object):
    """Same functionality as NumericalThetaCalculator, but uses
    ``scipy.integrate.quad`` and ``scipy.optimize.fsolve`` instead.

    Notes:
    It is more precise than ThetaCalculator and doesn't
    require you to think about which theta binning and range
    gives your desired precision.
    If you have to compute many thetas this can be quite slow
    because it is a root finding with nested integration.

    Parameters
    ----------
    dist : callable
        Probability distribution (probability per theta ^ 2)
    theta_max : float
        Integration range will be 0 .. theta_max ^ 2
    normalize : bool
        Normalize discretized distribution to 1?
    """
    def __init__(self, dist, theta_max, normalize=False):
        self.dist = dist
        self.theta_max = theta_max
        if normalize:
            self.p_total = self.containment_fraction(theta_max, False)
        else:
            self.p_total = 1

    def containment_fraction(self, theta):
        from scipy.integrate import quad
        p = quad(self.dist, 0, theta ** 2)[0]
        return p / self.p_total

    def containment_radius(self, containment_fraction):
        """Compute containment angle using the containment_fraction
        method plus numerical root finding."""
        from scipy.optimize import brentq

        def f(theta):
            return self.containment_fraction(theta) - containment_fraction
        return brentq(f, 0, self.theta_max)


class ThetaCalculator2D(ThetaCalculatorScipy):
    """Methods to compute theta and containment
    for a given 2D probability distribution image.

    Typically this method is used for PSF-convolved
    model images, where analytical distributions or
    1D distributions are not available.

    @note: The theta and containment is calculated relative
    to the origin (x, y) = (0, 0).

    @note: We do simple bin summing. In principle we could
    do integration over bins by using scipy.integrate.dblquad
    in combination with e.g. scipy.interpolate.interp2d,
    but for the speed / precision we need this is overkill.

    @todo: I just realized that probably the best thing to
    do is to bin (x,y) -> theta2, make a spline interpolation
    and then call ThetaCalculatorScipy!

    Parameters
    ----------
    dist : 2-dimensional array
        Probability distribution (per dx * dy)
    x : 2-dimensional array
        Pixel ``x`` coordinate array. Must match shape of ``dist``.
    x : 2-dimensional array
        Pixel ``x`` coordinate array. Must match share of ``dist``.
    """
    def __init__(self, dist, x, y):
        self.dist = dist / dist.sum()
        self.theta2 = x ** 2 + y ** 2
        self.theta_max = np.sqrt(self.theta2.max())

    def containment_fraction(self, theta):
        mask = self.theta2 < theta ** 2
        return self.dist[mask].sum()


class ModelThetaCalculator(ThetaCalculator):
    """Compute containment radius for given radially symmetric
    source and psf as well as desired containment fraction.

    Uses 2D images for the computation.
    Slow but simple, so useful to check more complicated methods.

    Source and PSF must be callable and return
    dP/dtheta (@todo: or dP/dtheta^2?)

    fov = field of view (deg)
    binsz = bin size (deg)

    The source is supposed to be contained in the FOV
    even after PSF convolution.
    """
    def __init__(self, source, psf, fov, binsz, call2d=False):
        from scipy.ndimage import convolve
        # Compute source and psf 2D images
        y, x = np.mgrid[-fov:fov:binsz, -fov:fov:binsz]
        theta2 = x * x + y * y
        if call2d:
            source_image = source(x, y)
            psf_image = psf(x, y)
        else:
            source_image = source.dpdtheta2(theta2)
            psf_image = psf.dpdtheta2(theta2)
        # Compute convolved image and normalize it
        p = convolve(source_image, psf_image)
        p /= p.sum()
        # Store the theta2 and p arrays
        self.p, self.theta2 = p, theta2

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Galactic radial source distribution probability density functions.

Attributes
----------
radial_distributions : `~astropy.utils.compat.odict.OrderedDict`
    Dictionary of available spatial distributions.

    Useful for automatic processing.

    def plot_spatial():
    import matplotlib.pyplot as plt
    max_radius = 20  # kpc
    r = np.linspace(0, max_radius, 100)
    plt.plot(r, normalize(density(radial_distributions['P90']), 0, max_radius)(r),
             color='b', linestyle='-', label='Paczynski 1990')
    plt.plot(r, normalize(density(radial_distributions['CB98']), 0, max_radius)(r),
             color='r', linestyle='--', label='Case&Battacharya 1998')
    plt.plot(r, normalize(density(radial_distributions['YK04']), 0, max_radius)(r),
             color='g', linestyle='-.', label='Yusifov&Kucuk 2004')
    plt.plot(r, normalize(density(radial_distributions['F06']), 0, max_radius)(r),
             color='m', linestyle='-', label='Faucher&Kaspi 2006')
    plt.plot(r, normalize(density(radial_distributions['L06']), 0, max_radius)(r),
             color='k', linestyle=':', label='Lorimer 2006')
    plt.xlim(0, max_radius)
    plt.ylim(0, 0.28)
    plt.xlabel('Galactocentric Distance [kpc]')
    plt.ylabel('Surface Density')
    plt.title('Comparison Radial Distribution Models (Surface Density)')
    plt.legend(prop={'size': 10})
    # plt.show()
"""
from __future__ import print_function, division
import numpy as np
from numpy.random import random_integers, uniform, normal

from numpy import exp, pi, log, abs, cos, sin
from astropy.utils.compat.odict import OrderedDict
from astropy.modeling import Fittable1DModel, Parameter

from ...utils.const import d_sun_to_galactic_center

__all__ = ['CaseBattacharya1998', 'FaucherKaspi2006', 'Lorimer2006',
           'Paczynski1990', 'YusifovKucuk2004', 'YusifovKucuk2004B',
           'Exponential', 'LogSpiral', 'FaucherSpiral', 'ValleeSpiral',
           'r_range', 'z_range',
           'radial_distributions',
           ]

R_SUN_GALACTIC = d_sun_to_galactic_center.value

# Simulation range used for random number drawing
r_range = 20  # kpc
z_range = 0.5  # kpc


class ProbabilityDensity1D(object):
    def __init__(self, function, x_min, x_max):
        if isinstance(function, str):
            self.function = None
        self.min = x_min
        self.max = x_max

    def normalize(self):
        pass

    def draw(self, N):
        pass


class Paczynski1990(Fittable1DModel):
    """
    Radial Birth Distribution of neutron stars - Paczynski 1990.

        .. math ::
            f(r)  = r r_{0} ^ (-2) \exp(-r / r_{0})

    Reference: http://adsabs.harvard.edu/abs/1990ApJ...348..485P

    Parameters
    ----------
    r_0 : float
        See formula

    See Also
    --------
    CaseBattacharya1998, YusifovKucuk2004, Lorimer2006, YusifovKucuk2004B,
    FaucherKaspi2006, Exponential
    """
    r_0 = Parameter(default=4.5)
    evolved = False

    def __init__(self, r_0=4.5, **kwargs):
        super(Paczynski1990, self).__init__(r_0=r_0, **kwargs)

    @staticmethod
    def eval(r, r_0):
        """One dimensional Paczynski 1990 model function"""
        return r * r_0 ** -2 * np.exp(-r / r_0)


class CaseBattacharya1998(Fittable1DModel):
    """Distribution of supernova remnants - Case and Battacharya 1998.

    .. math ::
        f(r) = r (r / r_{sun}) ^ a \exp(-b (r - r_{sun}) / r_{sun})

    Reference: http://adsabs.harvard.edu//abs/1998ApJ...504..761C

    Parameters
    ----------
    a : float
        See model formula
    b : float
        See model formula

    See Also
    --------
    Paczynski1990, YusifovKucuk2004, Lorimer2006, YusifovKucuk2004B,
    FaucherKaspi2006, Exponential
    """
    a = Parameter(default=2)
    b = Parameter(default=3.53)
    evolved = True

    def __init__(self, a=2, b=3.53, **kwargs):
        super(CaseBattacharya1998, self).__init__(a=a, b=b, **kwargs)

    @staticmethod
    def eval(r, a, b):
        """One dimensional Case and Battacharya model function"""
        term1 = r * (r / R_SUN_GALACTIC) ** a
        term2 = np.exp(-b * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)
        return term1 * term2


class YusifovKucuk2004(Fittable1DModel):
    """Evolved pulsar distribution - Yusifov and Kucuk 2004.

    .. math ::
        f(r) = TODO

    Used by Faucher-Guigere and Kaspi.
    Density at ``r = 0`` is nonzero.

    Reference: http://adsabs.harvard.edu/abs/2004A%26A...422..545Y

    Parameters
    ----------
    a : float
        See model formula
    b : float
        See model formula
    r_1 : float
        See model formula

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, Lorimer2006, YusifovKucuk2004B,
    FaucherKaspi2006, Exponential
    """
    a = Parameter(default=1.64)
    b = Parameter(default=4.01)
    r_1 = Parameter(default=0.55)
    evolved = True

    def __init__(self, a=1.64, b=4.01, r_1=0.55, **kwargs):
        super(YusifovKucuk2004, self).__init__(a=a, b=b, r_1=r_1, **kwargs)

    @staticmethod
    def eval(r, a, b, r_1):
        """One dimensional Yusifov Kucuk model function"""
        term1 = r * ((r + r_1) / (R_SUN_GALACTIC + r_1)) ** a
        term2 = np.exp(-b * (r - R_SUN_GALACTIC) / (R_SUN_GALACTIC + r_1))
        return term1 * term2


class YusifovKucuk2004B(Fittable1DModel):
    """Birth pulsar distribution - Yusifov & Kucuk 2004.

    .. math ::
        f(r) = (r / r_{sun}) ^ a \exp(-b (r / r_{sun}))

    Derived empirically from OB-stars distribution.

    Reference: http://adsabs.harvard.edu/abs/2004A%26A...422..545Y

    Parameters
    ----------
    a : float
        See model formula
    b : float
        See model formula

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    FaucherKaspi2006, Exponential
    """
    a = Parameter(default=4)
    b = Parameter(default=6.8)
    evolved = False

    def __init__(self, a=4, b=6.8, **kwargs):
        super(YusifovKucuk2004B, self).__init__(a=a, b=b, **kwargs)

    @staticmethod
    def eval(r, a, b):
        """One dimensional Yusifov Kucuk model function"""
        return (r / R_SUN_GALACTIC) ** a * np.exp(-b * (r / R_SUN_GALACTIC))


class FaucherKaspi2006(Fittable1DModel):
    """Displaced Gaussian distribution - Faucher-Giguere & Kaspi 2006.

    .. math ::
        f(r) = 1 / \sqrt(2 \pi \sigma) \exp(-\frac{(r - R_0)^2}{2 \sigma ^ 2})

    Proposed as a pulsar birth distribution in Appendix B.

    Parameters
    ----------
    r_0 : float
        See model formula
    sigma : float
        See model formula

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    YusifovKucuk2004B, Exponential
    """
    r_0 = Parameter()
    sigma = Parameter()
    evolved = False

    def __init__(self, r_0=7.04, sigma=1.83, **kwargs):
        super(FaucherKaspi2006, self).__init__(r_0=r_0, sigma=sigma, **kwargs)

    @staticmethod
    def eval(r, r_0, sigma):
        """One dimensional Faucher-Giguere and Kaspi model function"""
        term1 = 1. / np.sqrt(2 * pi * sigma)
        term2 = np.exp(-(r - r_0) ** 2 / (2 * sigma ** 2))
        return term1 * term2


class Lorimer2006(Fittable1DModel):
    """Evolved pulsar distribution - Lorimer 2006.

    .. math ::
        f(r) = r (r / r_{sun}) ^ a \exp(-b (r - r_{sun}) / r_sun)

    Surface density using the NE2001 Model.
    Similar to Kucuk, but core density is zero.

    Parameters
    ----------
    r : array_like
        Galactic radius (kpc)
    a, b : array_like
        See formula

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    YusifovKucuk2004B, FaucherKaspi2006

    """
    a = Parameter()
    b = Parameter()
    evolved = True

    def __init__(self, a=1.9, b=5.0, **kwargs):
        super(Lorimer2006, self).__init__(a=a, b=b, **kwargs)

    @staticmethod
    def eval(r, a, b):
        """Radial density function Lorimer 2006"""
        term1 = r * (r / R_SUN_GALACTIC) ** a
        term2 = np.exp(-b * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)
        return term1 * term2


class Exponential(Fittable1DModel):
    """
    Exponential distribution.

    .. math ::
        f(z) = \exp(-|z| / z_0)

    Usually used for height distribution above the Galactic plane,
    with 0.05 kpc as a commonly used birth height distribution.

    Parameters
    ----------
    z_0 : float
        Scale height of the distribution

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    YusifovKucuk2004B, FaucherKaspi2006, Exponential
    """
    z_0 = Parameter(default=0.05)
    evolved = False

    def __init__(self, z_0=0.05, **kwargs):
        super(Exponential, self).__init__(z_0=z_0, **kwargs)

    @staticmethod
    def eval(z, z_0):
        """One dimensional exponential model function"""
        return np.exp(-np.abs(z) / z_0)


class LogSpiral(object):
    r"""Logarithmic spiral.

    Reference: http://en.wikipedia.org/wiki/Logarithmic_spiral
    """

    def xy_position(self, theta=None, radius=None, spiralarm_index=0):
        """Compute (x, y) position for a given angle or radius.

        Parameters
        ----------
        theta : array_like
            Angle (deg)
        radius : array_like
            Radius (kpc)
        spiralarm_index : int
            Spiral arm index

        Returns
        -------
        x, y : array_like
            Position (x, y)
        """
        if (theta == None) and not (radius == None):
            theta = self.theta(radius, spiralarm_index=spiralarm_index)
        elif (radius == None) and not (theta == None):
            radius = self.radius(theta, spiralarm_index=spiralarm_index)
        else:
            ValueError('Specify only one of: theta, radius')

        theta = np.radians(theta)
        x = radius * cos(theta)
        y = radius * sin(theta)
        return x, y

    def radius(self, theta, spiralarm_index):
        """Radius for a given angle.

        Parameters
        ----------
        theta : array_like
            Angle (deg)
        spiralarm_index : int
            Spiral arm index

        Returns
        -------
        radius : array_like
            Radius (kpc)
        """
        k = self.k[spiralarm_index]
        r_0 = self.r_0[spiralarm_index]
        theta_0 = self.theta_0[spiralarm_index]
        d_theta = np.radians(theta - theta_0)
        radius = r_0 * exp(d_theta / k)
        return radius

    def theta(self, radius, spiralarm_index):
        """Angle for a given radius.

        Parameters
        ----------
        radius : array_like
            Radius (kpc)
        spiralarm_index : int
            Spiral arm index

        Returns
        -------
        theta : array_like
            Angle (deg)
        """
        k = self.k[spiralarm_index]
        r_0 = self.r_0[spiralarm_index]
        theta_0 = self.theta_0[spiralarm_index]
        theta_0 = np.radians(theta_0)
        theta = k * log(radius / r_0) + theta_0
        return np.degrees(theta)


class FaucherSpiral(LogSpiral):
    r"""Milky way spiral arm model from Faucher et al. (2006).

    Reference: http://adsabs.harvard.edu/abs/2006ApJ...643..332F
    """
    # Parameters
    k = np.array([4.25, 4.25, 4.89, 4.89])
    r_0 = np.array([3.48, 3.48, 4.9, 4.9])  # kpc
    theta_0 = np.array([1.57, 4.71, 4.09, 0.95])  # rad
    spiralarms = np.array(['Norma', 'Carina Sagittarius', 'Perseus', 'Crux Scutum'])

    def __call__(self, radius, blur=True, stdv=0.07, r_exp=2.857):
        """Draw random position from spiral arm distribution.

        Returns the corresponding angle theta[rad] to a given radius[kpc] and number of spiralarm.
        Possible numbers are:
        Norma = 0,
        Carina Sagittarius = 1,
        Perseus = 2
        Crux Scutum = 3.
        Returns dx and dy, if blurring= true."""
        N = random_integers(0, 3, radius.size)  # Choose Spiralarm
        theta = self.k[N] * log(radius / self.r_0[N]) + self.theta_0[N]  # Compute angle
        spiralarm = self.spiralarms[N]  # List that contains in wich spiralarm a postion lies

        if blur:  # Apply blurring model according to Faucher
            dr = abs(normal(0, stdv * radius, radius.size))
            dtheta = uniform(0, 2 * pi, radius.size)
            dx = dr * cos(dtheta)
            dy = dr * sin(dtheta)
            theta = theta + dtheta * exp(-radius / r_exp)

        return theta, spiralarm, dx, dy


class ValleeSpiral(LogSpiral):
    r"""Milky way spiral arm model from Vallee (2008).

    Reference: http://adsabs.harvard.edu/abs/2008AJ....135.1301V
    """
    # Model parameters
    p = 12.8  # pitch angle in deg
    m = 4  # number of spiral arms
    r_sun = 7.6  # distance sun to Galactic center in kpc
    r_0 = 2.1  # spiral inner radius in kpc
    theta_0 = -20  # Norma spiral arm start angle

    spiralarms = np.array(['Norma', 'Carina Sagittarius', 'Perseus', 'Crux Scutum'])

    def __init__(self):
        self.r_0 = self.r_0 * np.ones(4)
        self.theta_0 = self.theta_0 + np.array([0, 90, 180, 270])
        self.k = 1. / np.tan(np.radians(self.p)) * np.ones(4)

        # Compute start and end point of the bar
        x_0, y_0 = self.xy_position(radius=2.1, spiralarm_index=0)
        x_1, y_1 = self.xy_position(radius=2.1, spiralarm_index=2)
        self.bar = dict(x=np.array([x_0, x_1]), y=np.array([y_0, y_1]))

# TODO: this is not picked up in the HTML docs ... don't know why.
# http://sphinx-doc.org/latest/ext/example_numpy.html
# For now I add it in the module-level docstring in an `Attributes` section.
radial_distributions = OrderedDict()
"""Dictionary of available spatial distributions.

Useful for automatic processing.
"""
radial_distributions['P90'] = Paczynski1990
radial_distributions['CB98'] = CaseBattacharya1998
radial_distributions['YK04'] = YusifovKucuk2004
radial_distributions['YK04B'] = YusifovKucuk2004B
radial_distributions['F06'] = FaucherKaspi2006
radial_distributions['L06'] = Lorimer2006

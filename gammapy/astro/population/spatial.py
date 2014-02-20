# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Galactic radial source distribution probability density functions.
"""
from __future__ import print_function, division
import numpy as np
from numpy.random import random_integers, uniform, normal
from numpy import exp, sqrt, pi, log, abs, cos, sin

__all__ = ['CB98', 'F06', 'L06', 'P90', 'YK04', 'YK04B',
           'LogSpiral', 'FaucherSpiral', 'ValleeSpiral',
           'r_range', 'z_range']

R_SUN_GALACTIC = 8  # kpc

# Simulation range used for random number drawing
r_range = 20  # kpc
z_range = 0.5  # kpc


def P90(r, R0=4.5):
    r"""Radial Birth Distribution of neutron stars - Paczynski 1990.

    .. math ::
        f(r)  = r R_{0} ^ (-2) \exp(-r / R_{0})
    
    Reference: http://adsabs.harvard.edu/abs/1990ApJ...348..485P
    
    Parameters
    ----------
    r : array_like
        Galactic radius (kpc)
    R0 : array_like
        See formula
    
    Returns
    -------
    density : array_like
        Density in radius ``r``
    """
    return r * R0 ** -2 * exp(-r / R0)


def CB98(r, a=2, b=3.53):
    r"""Distribution of supernova remnants - Case and Battacharya 1998.
    
    .. math ::
        f(r) = r (r / r_{sun}) ^ a \exp(-b (r - r_{sun}) / r_{sun})
    
    Reference: http://adsabs.harvard.edu//abs/1998ApJ...504..761C
    
    Parameters
    ----------
    r : array_like
        Galactic radius (kpc)
    a, b : array_like
        See formula

    Returns
    -------
    density : array_like
        Density in radius ``r``
    """
    term1 = r * (r / R_SUN_GALACTIC) ** a
    term2 = exp(-b * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)
    return term1 * term2


def YK04(r, a=1.64, b=4.01, R1=0.55):
    r"""Evolved pulsar distribution - Yusifov and Kucuk 2004.
    
    .. math ::
        f(r) = TODO
    
    Used by Faucher-Guigere and Kaspi.
    Density at ``r = 0`` is nonzero.

    Reference: http://adsabs.harvard.edu/abs/2004A%26A...422..545Y
    
    Parameters
    ----------
    r : array_like
        Galactic radius (kpc)
    a, b, R1 : array_like
        See formula

    Returns
    -------
    density : array_like
        Density in radius ``r``
    """
    term1 = r * ((r + R1) / (R_SUN_GALACTIC + R1)) ** a
    term2 = exp(-b * (r - R_SUN_GALACTIC) / (R_SUN_GALACTIC + R1))
    return term1 * term2


def YK04B(r, a=4, b=6.8):
    r"""Birth pulsar distribution - Yusifov & Kucuk 2004.
    
    .. math ::
        f(r) = (r / r_{sun}) ^ a \exp(-b (r / r_{sun}))
    
    Derived empirically from OB-stars distribution.

    Reference: http://adsabs.harvard.edu/abs/2004A%26A...422..545Y

    Parameters
    ----------
    r : array_like
        Galactic radius (kpc)
    a, b : array_like
        See formula

    Returns
    -------
    density : array_like
        Density in radius ``r``
    """
    return (r / R_SUN_GALACTIC) ** a * exp(-b * (r / R_SUN_GALACTIC))


def F06(r, R0=7.04, sigma=1.83):
    r"""Displaced Gaussian distribution - Faucher-Giguere & Kaspi 2006.
    
    .. math ::
        f(r) = 1 / \sqrt(2 \pi \sigma) \exp(-\frac{(r - R_0)^2}{2 \sigma ^ 2})

    Proposed as a pulsar birth distribution in Appendix B.
    
    Parameters
    ----------
    r : array_like
        Galactic radius (kpc)
    R0, sigma : array_like
        See formula

    Returns
    -------
    density : array_like
        Density in radius ``r``
    """
    term1 = 1. / sqrt(2 * pi * sigma)
    term2 = exp(-(r - R0) ** 2 / (2 * sigma ** 2))
    return term1 * term2 


def L06(r, a=1.9, b=5.0):
    r"""Evolved pulsar distribution - Lorimer 2006.
    
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

    Returns
    -------
    density : array_like
        Density in radius ``r``
    """
    term1 = r * (r / R_SUN_GALACTIC) ** a
    term2 = exp(-b * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)
    return term1 * term2


def exponential(z, z0=0.05):
    r"""Exponential distribution.

    .. math ::
        f(z) = \exp(-|z| / z_0)

    Usually used for height distribution above the Galactic plane,
    with 0.05 kpc as a commonly used birth height distribution.

    Parameters
    ----------
    z : array_like
        Galactic z-coordinate (kpc)
    Returns
    -------
    density : array_like
        Density in height ``z``    
    """
    return exp(-abs(z) / z0)


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

# Dictionary of available distributions.
# Useful for automatic processing.
distributions = {'P90': P90, 'CB98': CB98, 'YK04': YK04,
                 'YK04B': YK04B, 'F06': F06, 'L06': L06}

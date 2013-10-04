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
    """Radial Birth Distribution of NS proposed by Paczynski 1990."""
    return r * R0 ** -2 * exp(-r / R0)


def CB98(r, a=2, b=3.53):
    """Current Distribution of SNR proposed by
    Case and Battacharya 1998."""
    return r * (r / R_SUN_GALACTIC) ** a * exp(-b * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)


def YK04(r, a=1.64, b=4.01, R1=0.55):
    """Evolved Distribution of Pulsars proposed by Yusifov
    and Kucuk 2004. Used by Faucher-Guigere and Kaspi.
    Density at r=0 is nonzero."""
    return r * ((r + R1) / (R_SUN_GALACTIC + R1)) ** a * exp(-b * (r - R_SUN_GALACTIC) / (R_SUN_GALACTIC + R1))


def YK04B(r, a=4, b=6.8):
    """Birth distribution of Pulsars, proposed by Yusifov & Kucuk 2006.
    Derived empirically from OB-stars distribution."""
    return (r / R_SUN_GALACTIC) ** a * exp(-b * (r / R_SUN_GALACTIC))


def F06(r, R0=7.04, StDev=1.83):
    """Displaced Gaussian (Probability function), proposed by
    as a birth distribution.
    @see Faucher-Giguere & Kaspi 2006 Appendix B"""
    return 1. / sqrt(2 * pi * StDev) * exp(-(r - R0) ** 2 / (2 * StDev ** 2))


def L06(r, a=1.9, b=5.0):
    """Evolved Distribution (Surface Density) proposed by Lorimer 2006
    using the NE2001 Model. Similar to Kucuk, but core density is zero."""
    return r * (r / R_SUN_GALACTIC) ** a * exp(-b * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)


def exponential(z, z0=0.05):
    """Exponential distribution.
    Usually used for height distribution above the Galactic plane,
    with 0.05 kpc as a commonly used birth height distribution."""
    return exp(-abs(z) / z0)


class LogSpiral(object):
    """Logarithmic spiral
    Reference: http://en.wikipedia.org/wiki/Logarithmic_spiral
    """

    def xy_position(self, theta=None, radius=None, spiralarm_index=0):
        """Compute (x, y) position for a given angle `theta` (in deg) or `radius` (in kpc)
        and spiral arm index.
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
        """
        theta in deg
        radius in kpc
        """
        k = self.k[spiralarm_index]
        r_0 = self.r_0[spiralarm_index]
        theta_0 = self.theta_0[spiralarm_index]
        d_theta = np.radians(theta - theta_0)
        radius = r_0 * exp(d_theta / k)
        return radius

    def theta(self, radius, spiralarm_index):
        """
        radius in kpc
        theta in deg
        """
        k = self.k[spiralarm_index]
        r_0 = self.r_0[spiralarm_index]
        theta_0 = self.theta_0[spiralarm_index]
        theta_0 = np.radians(theta_0)
        theta = k * log(radius / r_0) + theta_0
        return np.degrees(theta)


class FaucherSpiral(LogSpiral):
    """Milky way spiral arm model from Faucher et al. (2006)
    
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
    """Milky way spiral arm model from Vallee (2008)
    
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

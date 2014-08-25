# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Galactic radial source distribution probability density functions.

Attributes
----------
radial_distributions : `~astropy.utils.compat.odict.OrderedDict`
    Dictionary of available spatial distributions.

    Useful for automatic processing.
"""
from __future__ import print_function, division
import numpy as np
from numpy.random import random_integers, uniform, normal
from numpy import exp, pi, log, abs, cos, sin
from astropy.units import Quantity
from astropy.utils.compat.odict import OrderedDict
from astropy.modeling import Fittable1DModel, Parameter
from ...utils.coordinates import cartesian, polar
from ...utils.const import d_sun_to_galactic_center

__all__ = ['CaseBattacharya1998', 'FaucherKaspi2006', 'Lorimer2006',
           'Paczynski1990', 'YusifovKucuk2004', 'YusifovKucuk2004B',
           'Exponential', 'LogSpiral', 'FaucherSpiral', 'ValleeSpiral',
           'radial_distributions',
           'RMIN', 'RMAX', 'ZMIN', 'ZMAX',
           ]

R_SUN_GALACTIC = d_sun_to_galactic_center.value

# Simulation range used for random number drawing
RMIN, RMAX = Quantity(0, 'kpc'), Quantity(20, 'kpc')
ZMIN, ZMAX = Quantity(-0.5, 'kpc'), Quantity(0.5, 'kpc')


class Paczynski1990(Fittable1DModel):
    """
    Radial distribution of the birth surface density of neutron stars - Paczynski 1990.

    .. math ::
        f(r) = A r_{exp}^{-2} \\exp \\left(-\\frac{r}{r_{exp}} \\right)

    Reference: http://adsabs.harvard.edu/abs/1990ApJ...348..485P (Formula (2))

    Parameters
    ----------
    amplitude : float
        See formula
    r_exp : float
        See formula

    See Also
    --------
    CaseBattacharya1998, YusifovKucuk2004, Lorimer2006, YusifovKucuk2004B,
    FaucherKaspi2006, Exponential
    """
    amplitude = Parameter()
    r_exp = Parameter()
    evolved = False

    def __init__(self, amplitude=1, r_exp=4.5, **kwargs):
        super(Paczynski1990, self).__init__(amplitude=amplitude,
                                            r_exp=r_exp, **kwargs)

    @staticmethod
    def eval(r, amplitude, r_exp):
        """One dimensional Paczynski 1990 model function"""
        return amplitude * r_exp ** -2 * np.exp(-r / r_exp)


class CaseBattacharya1998(Fittable1DModel):
    """
    Radial distribution of the surface density of supernova remnants in the
    galaxy - Case & Battacharya 1998.

    .. math ::
        f(r) = A \\left( \\frac{r}{r_{\\odot}} \\right) ^ \\alpha \\exp 
        \\left[ -\\beta \\left( \\frac{ r - r_{\\odot}}{r_{\\odot}} \\right) \\right]

    Reference: http://adsabs.harvard.edu//abs/1998ApJ...504..761C (Formula (14))

    Parameters
    ----------
    amplitude : float
        See model formula
    alpha : float
        See model formula
    beta : float
        See model formula

    See Also
    --------
    Paczynski1990, YusifovKucuk2004, Lorimer2006, YusifovKucuk2004B,
    FaucherKaspi2006, Exponential
    """
    amplitude = Parameter()
    alpha = Parameter()
    beta = Parameter()
    evolved = True

    def __init__(self, amplitude=1., alpha=2, beta=3.53, **kwargs):
        super(CaseBattacharya1998, self).__init__(amplitude=amplitude,
                                                  alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def eval(r, amplitude, alpha, beta):
        """One dimensional Case & Battacharya 2006 model function"""
        term1 = (r / R_SUN_GALACTIC) ** alpha
        term2 = np.exp(-beta * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)
        return amplitude * term1 * term2


class YusifovKucuk2004(Fittable1DModel):
    """
    Radial distribution of the surface density of pulsars in the galaxy - Yusifov & Kucuk 2004.

    .. math ::
        f(r) = A \\left ( \\frac{r + r_1}{r_{\\odot} + r_1} \\right )^a \\exp 
        \\left [-b \\left( \\frac{r - r_{\\odot}}{r_{\\odot} + r_1} \\right ) \\right ]

    Used by Faucher-Guigere and Kaspi. Density at ``r = 0`` is nonzero.

    Reference: http://adsabs.harvard.edu/abs/2004A%26A...422..545Y (Formula (15))

    Parameters
    ----------
    amplitude : float
        See model formula
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
    amplitude = Parameter()
    a = Parameter()
    b = Parameter()
    r_1 = Parameter()
    evolved = True

    def __init__(self, amplitude=1, a=1.64, b=4.01, r_1=0.55, **kwargs):
        super(YusifovKucuk2004, self).__init__(amplitude=amplitude,
                                               a=a, b=b, r_1=r_1, **kwargs)

    @staticmethod
    def eval(r, amplitude, a, b, r_1):
        """One dimensional Yusifov & Kucuk 2004 model function"""
        term1 = ((r + r_1) / (R_SUN_GALACTIC + r_1)) ** a
        term2 = np.exp(-b * (r - R_SUN_GALACTIC) / (R_SUN_GALACTIC + r_1))
        return amplitude * term1 * term2


class YusifovKucuk2004B(Fittable1DModel):
    """
    Radial distribution of the surface density of OB stars in the galaxy - Yusifov & Kucuk 2004.

    .. math ::
        f(r) = A \\left( \\frac{r}{r_{\\odot}} \\right) ^ a
        \\exp \\left[ -b \\left( \\frac{r}{r_{\\odot}} \\right) \\right]

    Derived empirically from OB-stars distribution.

    Reference: http://adsabs.harvard.edu/abs/2004A%26A...422..545Y (Formula (17))

    Parameters
    ----------
    amplitude : float
        See model formula
    a : float
        See model formula
    b : float
        See model formula

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    FaucherKaspi2006, Exponential
    """
    amplitude = Parameter()
    a = Parameter()
    b = Parameter()
    evolved = False

    def __init__(self, amplitude=1, a=4, b=6.8, **kwargs):
        super(YusifovKucuk2004B, self).__init__(amplitude=amplitude,
                                                a=a, b=b, **kwargs)

    @staticmethod
    def eval(r, amplitude, a, b):
        """One dimensional Yusifov & Kucuk 2004 model function"""
        return amplitude * (r / R_SUN_GALACTIC) ** a * np.exp(-b * (r / R_SUN_GALACTIC))


class FaucherKaspi2006(Fittable1DModel):
    """
    Radial distribution of the birth surface density of pulsars in the galaxy - Faucher-Giguere & Kaspi 2006.

    .. math ::
        f(r) = A \\frac{1}{\\sqrt{2 \pi} \sigma} \\exp 
        \\left(- \\frac{(r - r_0)^2}{2 \sigma ^ 2}\\right)

    Reference: http://adsabs.harvard.edu/abs/2006ApJ...643..332F (Appendix B)

    Parameters
    ----------
    amplitude : float
        See model formula
    r_0 : float
        See model formula
    sigma : float
        See model formula

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    YusifovKucuk2004B, Exponential
    """
    amplitude = Parameter()
    r_0 = Parameter()
    sigma = Parameter()
    evolved = False

    def __init__(self, amplitude=1, r_0=7.04, sigma=1.83, **kwargs):
        super(FaucherKaspi2006, self).__init__(amplitude=amplitude,
                                               r_0=r_0, sigma=sigma, **kwargs)

    @staticmethod
    def eval(r, amplitude, r_0, sigma):
        """One dimensional Faucher-Giguere & Kaspi 2006 model function"""
        term1 = 1. / np.sqrt(2 * pi * sigma)
        term2 = np.exp(-(r - r_0) ** 2 / (2 * sigma ** 2))
        return amplitude * term1 * term2


class Lorimer2006(Fittable1DModel):
    """
    Radial distribution of the suface density of pulsars in the galaxy - Lorimer 2006.

    .. math ::
        f(r) = A \\left( \\frac{r}{r_{\\odot}} \\right) ^ B \\exp 
        \\left[ -C \\left( \\frac{r - r_{\\odot}}{r_{\\odot}} \\right) \\right]

    Reference: http://adsabs.harvard.edu/abs/2006MNRAS.372..777L (Formula (10))

    Parameters
    ----------
    amplitude : float
        See model formula
    B : float
        See model formula
    C : float
        See model formula

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    YusifovKucuk2004B, FaucherKaspi2006

    """
    amplitude = Parameter()
    B = Parameter()
    C = Parameter()
    evolved = True

    def __init__(self, amplitude=1, B=1.9, C=5.0, **kwargs):
        super(Lorimer2006, self).__init__(amplitude=amplitude, B=B, C=C, **kwargs)

    @staticmethod
    def eval(r, amplitude, B, C):
        """One dimensional Lorimer 2006 model function"""
        term1 = (r / R_SUN_GALACTIC) ** B
        term2 = np.exp(-C * (r - R_SUN_GALACTIC) / R_SUN_GALACTIC)
        return amplitude * term1 * term2


class Exponential(Fittable1DModel):
    """
    Exponential distribution.

    .. math ::
        f(z) = A \\exp \\left(- \\frac{|z|}{z_0} \\right)

    Usually used for height distribution above the Galactic plane,
    with 0.05 kpc as a commonly used birth height distribution.

    Parameters
    ----------
    amplitude : float
        See model formula
    z_0 : float
        Scale height of the distribution

    See Also
    --------
    CaseBattacharya1998, Paczynski1990, YusifovKucuk2004, Lorimer2006,
    YusifovKucuk2004B, FaucherKaspi2006, Exponential
    """
    amplitude = Parameter()
    z_0 = Parameter()
    evolved = False

    def __init__(self, amplitude=1, z_0=0.05, **kwargs):
        super(Exponential, self).__init__(amplitude=amplitude, z_0=z_0, **kwargs)

    @staticmethod
    def eval(z, amplitude, z_0):
        """One dimensional exponential model function"""
        return amplitude * np.exp(-np.abs(z) / z_0)


class LogSpiral(object):
    """
    Logarithmic spiral.

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
    r"""Milky way spiral arm used in Faucher et al. (2006).

    Reference: http://adsabs.harvard.edu/abs/2006ApJ...643..332F
    """
    # Parameters
    k = Quantity([4.25, 4.25, 4.89, 4.89], 'rad')
    r_0 = Quantity([3.48, 3.48, 4.9, 4.9], 'kpc')
    theta_0 = Quantity([1.57, 4.71, 4.09, 0.95], 'rad')
    spiralarms = np.array(['Norma', 'Carina Sagittarius', 'Perseus', 'Crux Scutum'])

    def _blur(self, radius, theta, amount=0.07):
        """
        Blur the positions around the centroid of the spiralarm.

        The given positions are blurred by drawing a displacement in radius from
        a normal distribution, with sigma = amount * radius. And a direction
        theta from a uniform distribution in the interval [0, 2 * pi].

        Parameters
        ----------
        radius : `~astropy.units.Quantity`
            Radius coordinate
        theta : `~astropy.units.Quantity`
            Angle coordinate
        amount: float
            Amount of blurring of the position, given as a fraction of `radius`.
        """
        dr = Quantity(abs(normal(0, amount * radius, radius.size)), 'kpc')
        dtheta = Quantity(uniform(0, 2 * np.pi, radius.size), 'rad')
        x, y = cartesian(radius, theta)
        dx, dy = cartesian(dr, dtheta)
        return polar(x + dx, y + dy)

    def _gc_correction(self, radius, theta, r_corr=Quantity(2.857, 'kpc')):
        """
        Correction of source distribution towards the galactic center.

        To avoid spiralarm features near the Galactic Center, the position angle theta
        is blurred by a certain amount towards the GC.

        Parameters
        ----------
        radius : `~astropy.units.Quantity`
            Radius coordinate
        theta : `~astropy.units.Quantity`
            Angle coordinate
        r_corr : `~astropy.units.Quantity`
            Scale of the correction towards the GC
        """
        theta_corr = Quantity(uniform(0, 2 * pi, radius.size), 'rad')
        return radius, theta + theta_corr * np.exp(-radius / r_corr)

    def __call__(self, radius, blur=True):
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
            radius, theta = self._blur(radius, theta)
            radius, theta = self._gc_correction(radius, theta)
        return radius, theta, spiralarm


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
radial_distributions['CB98'] = CaseBattacharya1998
radial_distributions['F06'] = FaucherKaspi2006
radial_distributions['L06'] = Lorimer2006
radial_distributions['P90'] = Paczynski1990
radial_distributions['YK04'] = YusifovKucuk2004
radial_distributions['YK04B'] = YusifovKucuk2004B

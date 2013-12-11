# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from copy import deepcopy
import numpy as np
from numpy import pi, exp, sqrt, log

__all__ = ['Gauss2D', 'MultiGauss2D']


class Gauss2D(object):
    """2D symmetric Gaussian PDF.
    
    Parameters
    ----------
    sigma : float
        Width.
    """
    def __init__(self, sigma=1):
        self.sigma = np.asarray(sigma, 'f')

    @property
    def sigma2(self):
        return self.sigma * self.sigma

    def __call__(self, x, y):
        """dp / (dx dy) at position (x, y).
        
        Reference: http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
        """
        x, y = np.asarray(x, 'f'), np.asarray(y, 'f')
        theta2 = x * x + y * y
        return (1 / (2 * pi * self.sigma2) * 
                exp(-0.5 * theta2 / self.sigma2))

    def dpdtheta2(self, theta2):
        """dp / dtheta2 at position theta2 = theta ^ 2.
        """
        theta2 = np.asarray(theta2, 'f')
        return (1 / (2 * self.sigma2) * 
                exp(-0.5 * theta2 / self.sigma2))

    def containment_fraction(self, theta):
        """Containment fraction for a given containment angle.
        """
        theta = np.asarray(theta, 'f')
        return 1 - exp(-0.5 * theta ** 2 / self.sigma2)

    def containment_radius(self, containment_fraction):
        """Containment angle for a given containment fraction.
        """
        containment_fraction = np.asarray(containment_fraction, 'f')
        return self.sigma * sqrt(-2 * log(1 - containment_fraction))

    def convolve(self, sigma):
        """Convolve with another Gaussian PDF of width sigma.
        """
        return deepcopy(self).convolve_me(sigma)

    def convolve_me(self, sigma):
        """Convolve this object, i.e. change its sigma.
        """
        sigma = np.asarray(sigma, 'f')
        self.sigma = sqrt(self.sigma2 + sigma ** 2)
        return self


class MultiGauss2D(object):
    """Sum of multiple 2D Gaussians.

    @note This sum is no longer a PDF, it is not normalized to 1.
    @note The "norm" of each component represents the 2D integral,
    not the amplitude at the origin.
    """
    def __init__(self, sigmas, norms=None):
        # If no norms are given, you have a PDF.
        sigmas = np.asarray(sigmas, 'f')
        self.components = [Gauss2D(sigma) for sigma in sigmas]
        if norms is None:
            self.norms = np.ones(len(self.components))
        else:
            self.norms = np.asarray(norms, 'f')

    def __call__(self, x, y):
        x, y = np.asarray(x, 'f'), np.asarray(y, 'f')
        total = np.zeros_like(x)
        for norm, component in zip(self.norms, self.components):
            total += norm * component(x, y)
        return total

    @property
    def n_components(self):
        return len(self.components)

    @property
    def sigmas(self):
        return np.array([_.sigma for _ in self.components])

    @property
    def integral(self):
        # self.norms.sum()
        return np.nansum(self.norms)

    @property
    def amplitude(self):
        return self.__call__(0, 0)

    @property
    def max_sigma(self):
        return self.sigmas.max()

    @property
    def eff_sigma(self):
        """Effective sigma if we naively were to replace the
        MultiGauss2D with one Gauss2D."""
        sigma2s = np.array([component.sigma2 for component in self.components])
        return np.sqrt(np.sum(self.norms * sigma2s))

    def dpdtheta2(self, theta2):
        # Actually this is only a PDF if sum(norms) == 1
        theta2 = np.asarray(theta2, 'f')
        total = np.zeros_like(theta2)
        for norm, component in zip(self.norms, self.components):
            total += norm * component.dpdtheta2(theta2)
        return total

    def normalize(self):
        self.norms /= self.integral
        return self

    def containment_fraction(self, theta):
        """Containment fraction.
        
        Parameters
        ----------
        theta : array_like
            Containment angle
        """
        theta = np.asarray(theta, 'f')
        total = np.zeros_like(theta)
        for norm, component in zip(self.norms, self.components):
            total += norm * component.containment_fraction(theta)
        return total

    def containment_radius(self, containment_fraction):
        """Containment angle.
        
        Parameters
        ----------
        fraction : float
            Containment fraction
        """
        # I had big problems with fsolve running into negative thetas.
        # So instead I'll find a theta_max myself so that theta
        # is in the interval [0, theta_max] and then use good ol brentq
        if not containment_fraction < self.integral:
            raise ValueError('containment_fraction = {0} not possible for integral = {1}'
                             ''.format(containment_fraction, self.integral))
        from scipy.optimize import brentq

        def f(theta):
            # positive if theta too large
            return self.containment_fraction(theta) - containment_fraction
        # @todo: if it is an array we have to loop by hand!
        # containment = np.asarray(containment, 'f')
        # Inital guess for theta
        theta_max = self.eff_sigma
        # Expand until we really find a theta_max
        while f(theta_max) < 0:
            theta_max *= 2
        return brentq(f, 0, theta_max)

    def match_sigma(self, containment_fraction):
        """Compute equivalent Gauss width.
        
        Find the sigma of a single-Gaussian distribution that
        approximates this one, such that theta matches for a given
        containment."""
        theta1 = self.containment_radius(containment_fraction)
        theta2 = Gauss2D(sigma=1).containment_radius(containment_fraction)
        return theta1 / theta2

    def convolve(self, sigma, norm=1):
        """Convolve with another Gauss.
        
        Compute new norms and sigmas of all the components such that
        the new distribution represents the convolved old distribution
        by a Gaussian of width sigma and then multiplied by norm.

        This MultiGauss2D is unchanged, a new one is created and returned.
        This is useful if you need to e.g. compute theta for one PSF
        and many sigmas."""
        return deepcopy(self).convolve_me(sigma, norm)

    def convolve_me(self, sigma, norm=1):
        """Convolve this object, i.e. change its sigmas and norms.
        """
        sigma = np.asarray(sigma, 'f')
        norm = np.asarray(norm, 'f')
        for ii in range(self.n_components):
            self.components[ii].convolve_me(sigma)
            self.norms[ii] *= norm
        return self

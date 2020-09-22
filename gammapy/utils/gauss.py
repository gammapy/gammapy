# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multi-Gaussian distribution utilities (Gammapy internal)."""
import numpy as np
import scipy.optimize


class Gauss2DPDF:
    """2D symmetric Gaussian PDF.

    Reference: http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case

    Parameters
    ----------
    sigma : float
        Gaussian width.
    """

    def __init__(self, sigma=1):
        self.sigma = np.asarray(sigma, np.float64)

    @property
    def _sigma2(self):
        """Sigma squared (float)"""
        return self.sigma * self.sigma

    @property
    def amplitude(self):
        """PDF amplitude at the center (float)"""
        return self.__call(0, 0)

    def __call__(self, x, y=0):
        """dp / (dx dy) at position (x, y)

        Parameters
        ----------
        x : `~numpy.ndarray`
            x coordinate
        y : `~numpy.ndarray`, optional
            y coordinate

        Returns
        -------
        dpdxdy : `~numpy.ndarray`
            dp / (dx dy)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        theta2 = x * x + y * y
        amplitude = 1 / (2 * np.pi * self._sigma2)
        exponent = -0.5 * theta2 / self._sigma2
        return amplitude * np.exp(exponent)

    def dpdtheta2(self, theta2):
        """dp / dtheta2 at position theta2 = theta ^ 2

        Parameters
        ----------
        theta2 : `~numpy.ndarray`
            Offset squared

        Returns
        -------
        dpdtheta2 : `~numpy.ndarray`
            dp / dtheta2
        """
        theta2 = np.asarray(theta2, dtype=np.float64)

        amplitude = 1 / (2 * self._sigma2)
        exponent = -0.5 * theta2 / self._sigma2
        return amplitude * np.exp(exponent)

    def containment_fraction(self, theta):
        """Containment fraction.

        Parameters
        ----------
        theta : `~numpy.ndarray`
            Offset

        Returns
        -------
        containment_fraction : `~numpy.ndarray`
            Containment fraction
        """
        theta = np.asarray(theta, dtype=np.float64)

        return 1 - np.exp(-0.5 * theta ** 2 / self._sigma2)

    def containment_radius(self, containment_fraction):
        """Containment angle for a given containment fraction.

        Parameters
        ----------
        containment_fraction : `~numpy.ndarray`
            Containment fraction

        Returns
        -------
        containment_radius : `~numpy.ndarray`
            Containment radius
        """
        containment_fraction = np.asarray(containment_fraction, dtype=np.float64)

        return self.sigma * np.sqrt(-2 * np.log(1 - containment_fraction))

    def gauss_convolve(self, sigma):
        """Convolve with another Gaussian 2D PDF.

        Parameters
        ----------
        sigma : `~numpy.ndarray` or float
            Gaussian width of the new Gaussian 2D PDF to covolve with.

        Returns
        -------
        gauss_convolve : `~gammapy.modeling.models.Gauss2DPDF`
            Convolution of both Gaussians.
        """
        sigma = np.asarray(sigma, dtype=np.float64)

        new_sigma = np.sqrt(self._sigma2 + sigma ** 2)
        return Gauss2DPDF(new_sigma)


class MultiGauss2D:
    """Sum of multiple 2D Gaussians.

    Parameters
    ----------
    sigmas : `~numpy.ndarray`
            widths of the Gaussians to add
    norms : `~numpy.ndarray`, optional
            normalizations of the Gaussians to add

    Notes
    -----
    * This sum is no longer a PDF, it is not normalized to 1.
    * The "norm" of each component represents the 2D integral,
      not the amplitude at the origin.
    """

    def __init__(self, sigmas, norms=None):
        # If no norms are given, you have a PDF.
        sigmas = np.asarray(sigmas, dtype=np.float64)
        self.components = [Gauss2DPDF(sigma) for sigma in sigmas]

        if norms is None:
            self.norms = np.ones(len(self.components))
        else:
            self.norms = np.asarray(norms, dtype=np.float64)

    def __call__(self, x, y=0):
        """dp / (dx dy) at position (x, y)

        Parameters
        ----------
        x : `~numpy.ndarray`
            x coordinate
        y : `~numpy.ndarray`, optional
            y coordinate

        Returns
        -------
        total : `~numpy.ndarray`
            dp / (dx dy)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        total = np.zeros_like(x)
        for norm, component in zip(self.norms, self.components):
            total += norm * component(x, y)
        return total

    @property
    def n_components(self):
        """Number of components (int)"""
        return len(self.components)

    @property
    def sigmas(self):
        """Array of Gaussian widths (`~numpy.ndarray`)"""
        return np.array([_.sigma for _ in self.components])

    @property
    def integral(self):
        """Integral as sum of norms (`~numpy.ndarray`)"""
        return np.nansum(self.norms)

    @property
    def amplitude(self):
        """Amplitude at the center (float)"""
        return self.__call__(0, 0)

    @property
    def max_sigma(self):
        """Largest Gaussian width (float)"""
        return self.sigmas.max()

    @property
    def eff_sigma(self):
        r"""Effective Gaussian width for single-Gauss approximation (float)

        Notes
        -----
        The effective Gaussian width is given by:

        .. math:: \sigma_\mathrm{eff} = \sqrt{\sum_i N_i \sigma_i^2}

        where ``N`` is normalization and ``sigma`` is width.

        """
        sigma2s = np.array([component._sigma2 for component in self.components])
        return np.sqrt(np.sum(self.norms * sigma2s))

    def dpdtheta2(self, theta2):
        """dp / dtheta2 at position theta2 = theta ^ 2

        Parameters
        ----------
        theta2 : `~numpy.ndarray`
            Offset squared

        Returns
        -------
        dpdtheta2 : `~numpy.ndarray`
            dp / dtheta2
        """
        # Actually this is only a PDF if sum(norms) == 1
        theta2 = np.asarray(theta2, dtype=np.float64)

        total = np.zeros_like(theta2)
        for norm, component in zip(self.norms, self.components):
            total += norm * component.dpdtheta2(theta2)
        return total

    def normalize(self):
        """Normalize function.

        Returns
        -------
        norm_multigauss : `~gammapy.modeling.models.MultiGauss2D`
           normalized function
        """
        sum = self.integral
        if sum != 0:
            self.norms /= sum
        return self

    def containment_fraction(self, theta):
        """Containment fraction.

        Parameters
        ----------
        theta : `~numpy.ndarray`
            Offset

        Returns
        -------
        containment_fraction : `~numpy.ndarray`
            Containment fraction
        """
        theta = np.asarray(theta, dtype=np.float64)

        total = np.zeros_like(theta)
        for norm, component in zip(self.norms, self.components):
            total += norm * component.containment_fraction(theta)

        return total

    def containment_radius(self, containment_fraction):
        """Containment angle for a given containment fraction.

        Parameters
        ----------
        containment_fraction : `~numpy.ndarray`
            Containment fraction

        Returns
        -------
        containment_radius : `~numpy.ndarray`
            Containment radius
        """
        theta_max = 1e3

        def f(theta):
            # positive if theta too large
            return self.containment_fraction(theta) - containment_fraction

        theta = scipy.optimize.brentq(f, a=0, b=theta_max)
        if np.allclose(theta, theta_max):
            theta = np.inf

        return theta

    def match_sigma(self, containment_fraction):
        """Compute equivalent Gauss width.

        Find the sigma of a single-Gaussian distribution that
        approximates this one, such that theta matches for a given
        containment.

        Parameters
        ----------
        containment_fraction : `~numpy.ndarray`
            Containment fraction

        Returns
        -------
        sigma : `~numpy.ndarray`
            Equivalent containment radius
        """
        theta1 = self.containment_radius(containment_fraction)
        theta2 = Gauss2DPDF(sigma=1).containment_radius(containment_fraction)
        return theta1 / theta2

    def gauss_convolve(self, sigma, norm=1):
        """Convolve with another Gauss.

        Compute new norms and sigmas of all the components such that
        the new distribution represents the convolved old distribution
        by a Gaussian of width sigma and then multiplied by norm.

        This MultiGauss2D is unchanged, a new one is created and returned.
        This is useful if you need to e.g. compute theta for one PSF
        and many sigmas.

        Parameters
        ----------
        sigma : `~numpy.ndarray` or float
            Gaussian width of the new Gaussian 2D PDF to covolve with.
        norm : `~numpy.ndarray` or float
            Normalization of the new Gaussian 2D PDF to covolve with.

        Returns
        -------
        new_multi_gauss_2d : `~gammapy.modeling.models.MultiGauss2D`
            Convolution as new MultiGauss2D
        """
        sigma = np.asarray(sigma, dtype=np.float64)
        norm = np.asarray(norm, dtype=np.float64)

        sigmas, norms = [], []
        for ii in range(self.n_components):
            sigmas.append(self.components[ii].gauss_convolve(sigma).sigma)
            norms.append(self.norms[ii] * norm)

        return MultiGauss2D(sigmas, norms)

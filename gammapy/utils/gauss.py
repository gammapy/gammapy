# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multi-Gaussian distribution utitities (Gammapy internal)."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__doctest_requires__ = {("gaussian_sum_moments"): ["uncertainties"]}


class Gauss2DPDF(object):
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
        gauss_convolve : `~gammapy.image.models.Gauss2DPDF`
            Convolution of both Gaussians.
        """
        sigma = np.asarray(sigma, dtype=np.float64)

        new_sigma = np.sqrt(self._sigma2 + sigma ** 2)
        return Gauss2DPDF(new_sigma)


class MultiGauss2D(object):
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
        norm_multigauss : `~gammapy.image.models.MultiGauss2D`
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
        # I had big problems with fsolve running into negative thetas.
        # So instead I'll find a theta_max myself so that theta
        # is in the interval [0, theta_max] and then use good ol brentq
        if not containment_fraction < self.integral:
            raise ValueError(
                "containment_fraction = {} not possible for integral = {}"
                "".format(containment_fraction, self.integral)
            )
        from scipy.optimize import brentq

        def f(theta):
            # positive if theta too large
            return self.containment_fraction(theta) - containment_fraction

        # TODO: if it is an array we have to loop by hand!
        # containment = np.asarray(containment, dtype=np.float64)
        # Inital guess for theta
        theta_max = self.eff_sigma
        # Expand until we really find a theta_max
        while f(theta_max) < 0:
            theta_max *= 2
        return brentq(f, a=0, b=theta_max)

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
        new_multi_gauss_2d : `~gammapy.image.models.MultiGauss2D`
            Convolution as new MultiGauss2D
        """
        sigma = np.asarray(sigma, dtype=np.float64)
        norm = np.asarray(norm, dtype=np.float64)

        sigmas, norms = [], []
        for ii in range(self.n_components):
            sigmas.append(self.components[ii].gauss_convolve(sigma).sigma)
            norms.append(self.norms[ii] * norm)

        return MultiGauss2D(sigmas, norms)


def gaussian_sum_moments(F, sigma, x, y, cov_matrix, shift=0.5):
    """Compute image moments with uncertainties for sum of Gaussians.

    The moments are computed analytically, the formulae are documented below.

    Calls ``uncertainties.correlated_values`` to propagate the errors.

    Parameters
    ----------
    F : array
        Integral norms of the Gaussian components.
    sigmas : array
        Widths of the Gaussian components.
    x : array
        x positions of the Gaussian components.
    y : array
        y positions of the Gaussian components.
    cov_matrix : array
        Covariance matrix of the parameters. The columns have to follow the order:
        [sigma_1, x_1, y_1, F_1, sigma_2, x_2, y_2, F_2, ..., sigma_N, x_N, y_N, F_N]
    shift : float (default = 0.5)
        Depending on where the image values are given, the grid has to be
        shifted. If the values are given at the center of the pixel
        shift = 0.5.

    Returns
    -------
    nominal_values : list
        List of image moment nominal values:
        [F_sum, x_sum, y_sum, x_sigma, y_sigma, sqrt(x_sigma * y_sigma)]

        All values are given in pixel coordinates.
    std_devs : list
        List of image moment standard deviations.

    Examples
    --------
    A simple example for an image consisting of three Gaussians
    with zero covariance matrix:

    >>> import numpy as np
    >>> from gammapy.utils.gauss import gaussian_sum_moments
    >>> cov_matrix = np.zeros((12, 12))
    >>> F = [100, 200, 300]
    >>> sigma = [15, 10, 5]
    >>> x = [100, 120, 70]
    >>> y = [100, 90, 120]
    >>> nominal_values, std_devs = gaussian_sum_moments(F, sigma, x, y, cov_matrix)

    Notes
    -----

    The 0th moment (total flux) is given by:

    .. math::
        F_{\\Sigma} = \\int_{-\\infty}^{\\infty}f_{\\Sigma}(x, y)dx dy =
        \\sum_i^N F_i

    The 1st moments (position) are given by:

    .. math::
        x_{\\Sigma} = \\frac{1}{F_{\\Sigma}} \\int_{-\\infty}^{\\infty}x
        f_{\\Sigma}(x, y)dx dy = \\frac{1}{F_{\\Sigma}}\\sum_i^N x_iF_i

        y_{\\Sigma} = \\frac{1}{F_{\\Sigma}} \\int_{-\\infty}^{\\infty}y
        f_{\\Sigma}(x, y)dx dy = \\frac{1}{F_{\\Sigma}}\\sum_i^N y_iF_i

    The 2nd moments (extension) are given by:

    .. math::
        \\sigma_{\\Sigma_x}^2 = \\frac{1}{F_{\\Sigma}} \\sum_i^N F_i
        \\cdot (\\sigma_i^2 + x_i^2) - x_{\\Sigma}^2

        \\sigma_{\\Sigma_y}^2 = \\frac{1}{F_{\\Sigma}} \\sum_i^N F_i
        \\cdot (\\sigma_i^2 + y_i^2) - y_{\\Sigma}^2

    """
    import uncertainties

    # Check input arrays
    if len(F) != len(sigma) or len(F) != len(x) or len(F) != len(x):
        raise Exception("Input arrays have to have the same size")

    # Order parameter values
    values = []
    for i in range(len(F)):
        values += [sigma[i], x[i], y[i], F[i]]

    # Set up parameters with uncertainties
    parameters = uncertainties.correlated_values(values, cov_matrix)

    # Reorder parameters by splitting into 4-tuples
    parameters = list(zip(*[iter(parameters)] * 4))

    def zero_moment(parameters):
        """0th moment of the sum of Gaussian components."""
        F_sum = 0
        for component in parameters:
            F_sum += component[3]
        return F_sum

    def first_moment(parameters, F_sum, shift):
        """1st moment of the sum of Gaussian components."""
        x_sum, y_sum = 0, 0
        for component in parameters:
            x_sum += (component[1] + shift) * component[3]
            y_sum += (component[2] + shift) * component[3]
        return x_sum / F_sum, y_sum / F_sum

    def second_moment(parameters, F_sum, x_sum, y_sum, shift):
        """2nd moment of the sum of Gaussian components."""
        var_x_sum, var_y_sum = 0, 0
        for p in parameters:
            var_x_sum += ((p[1] + shift) ** 2 + p[0] ** 2) * p[3]
            var_y_sum += ((p[2] + shift) ** 2 + p[0] ** 2) * p[3]
        return var_x_sum / F_sum - x_sum ** 2, var_y_sum / F_sum - y_sum ** 2

    # Compute moments
    F_sum = zero_moment(parameters)
    x_sum, y_sum = first_moment(parameters, F_sum, shift)
    var_x_sum, var_y_sum = second_moment(parameters, F_sum, x_sum, y_sum, shift)

    # Return values and stddevs separately
    values = [
        F_sum,
        x_sum,
        y_sum,
        var_x_sum ** 0.5,
        var_y_sum ** 0.5,
        (var_x_sum * var_y_sum) ** 0.25,
    ]
    nominal_values = [_.nominal_value for _ in values]
    std_devs = [float(_.std_dev) for _ in values]

    return nominal_values, std_devs

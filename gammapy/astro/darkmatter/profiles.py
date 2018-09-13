# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter profiles."""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
import astropy.units as u
from ...extern import six
from ...utils.fitting import Parameter, Parameters
from ...spectrum.utils import integrate_spectrum

__all__ = [
    "DMProfile",
    "NFWProfile",
    "EinastoProfile",
    "IsothermalProfile",
    "BurkertProfile",
    "MooreProfile",
]


@six.add_metaclass(abc.ABCMeta)
class DMProfile(object):
    """DMProfile model base class."""

    LOCAL_DENSITY = 0.3 * u.GeV / (u.cm ** 3)
    """Local dark matter density as given in refenrece 2"""
    DISTANCE_GC = 8.33 * u.kpc
    """Distance to the Galactic Center as given in reference 2"""

    def __call__(self, radius):
        """Call evaluate method of derived classes."""
        kwargs = dict()
        for par in self.parameters.parameters:
            kwargs[par.name] = par.quantity

        return self.evaluate(radius, **kwargs)

    def scale_to_local_density(self):
        """Scale to local density."""
        scale = (self.LOCAL_DENSITY / self(self.DISTANCE_GC)).to("").value
        self.parameters["rho_s"].value *= scale

    def _eval_squared(self, radius):
        """Squared density at given radius."""
        return self(radius) ** 2

    def integral(self, rmin, rmax, **kwargs):
        r"""Integrate squared dark matter profile numerically.

        .. math::

            F(r_{min}, r_{max}) = \int_{r_{min}}^{r_{max}}\rho(r)^2 dr


        Parameters
        ----------
        rmin, rmax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to :func:`~gammapy.spectrum.integrate_spectrum`
        """
        integral = integrate_spectrum(self._eval_squared, rmin, rmax, **kwargs)
        return integral.to("GeV2 / cm5")


class NFWProfile(DMProfile):
    r"""NFW Profile.

    .. math::

        \rho(r) = \rho_s \frac{r_s}{r}\left(1 + \frac{r}{r_s}\right)^{-2}

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`
    rho_s : `~astropy.units.Quantity`
        Characteristic density, :math:`\rho_s`

    References
    ----------
    * `1997ApJ...490..493 <http://adsabs.harvard.edu/abs/1997ApJ...490..493N>`_
    * `2011JCAP...03..051 <http://adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    DEFAULT_SCALE_RADIUS = 24.42 * u.kpc
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, rho_s=1 * u.Unit("GeV / cm3")):
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else r_s
        self.parameters = Parameters(
            [Parameter("r_s", u.Quantity(r_s)), Parameter("rho_s", u.Quantity(rho_s))]
        )

    @staticmethod
    def evaluate(radius, r_s, rho_s):
        rr = radius / r_s
        return rho_s / (rr * (1 + rr) ** 2)


class EinastoProfile(DMProfile):
    r"""Einasto Profile.

    .. math::

        \rho(r) = \rho_s \exp{
            \left(-\frac{2}{\alpha}\left[
            \left(\frac{r}{r_s}\right)^{\alpha} - 1\right] \right)}

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`
    alpha : `~astropy.units.Quantity`
        :math:`\alpha`
    rho_s : `~astropy.units.Quantity`
        Characteristic density, :math:`\rho_s`

    References
    ----------
    * `1965TrAlm...5...87E <http://adsabs.harvard.edu/abs/1965TrAlm...5...87E>`_
    * `2011JCAP...03..051 <http://adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    DEFAULT_SCALE_RADIUS = 28.44 * u.kpc
    """Default scale radius as given in reference 2"""
    DEFAULT_ALPHA = 0.17
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, alpha=None, rho_s=1 * u.Unit("GeV / cm3")):
        alpha = self.DEFAULT_ALPHA if alpha is None else alpha
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else r_s

        self.parameters = Parameters(
            [
                Parameter("r_s", u.Quantity(r_s)),
                Parameter("alpha", u.Quantity(alpha)),
                Parameter("rho_s", u.Quantity(rho_s)),
            ]
        )

    @staticmethod
    def evaluate(radius, r_s, alpha, rho_s):
        rr = radius / r_s
        exponent = (2 / alpha) * (rr ** alpha - 1)
        return rho_s * np.exp(-1 * exponent)


class IsothermalProfile(DMProfile):
    r"""Isothermal Profile.

    .. math::

        \rho(r) = \frac{\rho_s}{1 + (r/r_s)^2}

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`

    References
    ----------
    * `1991MNRAS.249..523B <http://adsabs.harvard.edu/abs/1991MNRAS.249..523B>`_
    * `2011JCAP...03..051 <http://adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    DEFAULT_SCALE_RADIUS = 4.38 * u.kpc
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, rho_s=1 * u.Unit("GeV / cm3")):
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else r_s

        self.parameters = Parameters(
            [Parameter("r_s", u.Quantity(r_s)), Parameter("rho_s", u.Quantity(rho_s))]
        )

    @staticmethod
    def evaluate(radius, r_s, rho_s):
        rr = radius / r_s
        return rho_s / (1 + rr ** 2)


class BurkertProfile(DMProfile):
    r"""Burkert Profile.

    .. math::

        \rho(r) = \frac{\rho_s}{(1 + r/r_s)(1 + (r/r_s)^2)}

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`

    References
    ----------
    * `1995ApJ...447L..25B <http://adsabs.harvard.edu/abs/1995ApJ...447L..25B>`_
    * `2011JCAP...03..051 <http://adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    DEFAULT_SCALE_RADIUS = 12.67 * u.kpc
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, rho_s=1 * u.Unit("GeV / cm3")):
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else r_s

        self.parameters = Parameters(
            [Parameter("r_s", u.Quantity(r_s)), Parameter("rho_s", u.Quantity(rho_s))]
        )

    @staticmethod
    def evaluate(radius, r_s, rho_s):
        rr = radius / r_s
        return rho_s / ((1 + rr) * (1 + rr ** 2))


class MooreProfile(DMProfile):
    r"""Moore Profile.

    .. math::

        \rho(r) = \rho_s \left(\frac{r_s}{r}\right)^{1.16}
        \left(1 + \frac{r}{r_s} \right)^{-1.84}

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`

    References
    ----------
    * `2004MNRAS.353..624D <http://adsabs.harvard.edu/abs/2004MNRAS.353..624D>`_
    * `2011JCAP...03..051 <http://adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    DEFAULT_SCALE_RADIUS = 30.28 * u.kpc
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, rho_s=1 * u.Unit("GeV / cm3")):
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else r_s

        self.parameters = Parameters(
            [Parameter("r_s", u.Quantity(r_s)), Parameter("rho_s", u.Quantity(rho_s))]
        )

    @staticmethod
    def evaluate(radius, r_s, rho_s):
        rr = radius / r_s
        rr_ = r_s / radius
        return rho_s * rr_ ** 1.16 * (1 + rr) ** (-1.84)

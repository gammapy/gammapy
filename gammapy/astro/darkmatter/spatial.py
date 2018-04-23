# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Morphological dark matter models
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.modeling import Parameter, ParameterList
import astropy.units as u
import six, abc

__all__ = [
    'DMProfile',
    'NFWProfile',
    'EinastoProfile',
]

@six.add_metaclass(abc.ABCMeta)
class DMProfile(object):
    """DMProfile model base class.
    """
    LOCAL_DENSITY = 0.39 * u.GeV / (u.cm ** 3)
    """Local dark matter density"""
    DISTANCE_GC = 8.5 * u.kpc
    """Distance to the Galactic Center"""

    def __call__(self, radius):
        """Call evaluate method of derived classes"""
        kwargs = dict()
        for par in self.parameters.parameters:
            kwargs[par.name] = par.quantity

        return self.evaluate(radius, **kwargs)

    def scale_to_local_density(self):
        """Scale to local density"""
        scale = (self.LOCAL_DENSITY / self(self.DISTANCE_GC)).to('').value
        self.parameters['rho_s'].value *= scale


class NFWProfile(DMProfile):
    r"""NFW Profile.

    .. math::

        \rho(r) = \rho_s \left[
            \frac{r}{r_s}\left(1 + \frac{r}{r_s}\right)^2
            \right]^{-1}

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`
    rho_s : `~astropy.units.Quantity`
        Characteristic density, :math:`\rho_s`

    References
    ----------
    * `arXiv:astro-ph/9611107
    <https://arxiv.org/abs/astro-ph/9611107>`_
    * `arXiv:0908.0195
    <https://arxiv.org/abs/0908.0195>`_
    """
    DEFAULT_SCALE_RADIUS = 21 * u.kpc
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, rho_s=1*u.Unit('GeV / cm3')):
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else rs
        self.parameters = ParameterList([
            Parameter('r_s', u.Quantity(r_s)),
            Parameter('rho_s', u.Quantity(rho_s))
        ])

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
        math:`\alpha`
    rho_s : `~astropy.units.Quantity`
        Characteristic density, :math:`\rho_s`

    References
    ----------
    * `1965TrAlm...5...87E
    <http://adsabs.harvard.edu/abs/1965TrAlm...5...87E>`_
    * `arXiv:0908.0195
    <https://arxiv.org/abs/0908.0195>`_
    """
    DEFAULT_SCALE_RADIUS = 21 * u.kpc
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, rho_s=1*u.Unit('GeV / cm3')):
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else rs
        self.parameters = ParameterList([
            Parameter('r_s', u.Quantity(r_s)),
            Parameter('rho_s', u.Quantity(rho_s))
        ])

    @staticmethod
    def evaluate(radius, r_s, rho_s):
        rr = radius / r_s
        return rho_s / (rr * (1 + rr) ** 2)

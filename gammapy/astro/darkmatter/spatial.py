# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Morphological dark matter models
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.modeling import Parameter, ParameterList

__all__ = [
    'NFWProfile',
]


class NFWProfile(class):
    r"""NFW Profile.

    .. math::

        \frac{\rho(r)}{\rho_{\mathrm{crit}}} =
            \delta_c \left[
            \frac{r}{r_s}\left(1 + \frac{r}{r_s}\right)^2
            \right]^-1

    whhere :math:`\rho_{\mathrm{crit}}` is the critical density

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`
    delta_c : `~astropy.units.Quantity`
        Characteristic density, :math:`delta_c`

    References
    ----------
    * `arXiv:astro-ph/9611107
    <https://arxiv.org/abs/astro-ph/9611107>`_
    """

    def __init__(self, r_s, delta_c):
        self.parameters = ParameterList([
            Parameter('r_s', Quantity(r_s)),
            Parameter('delta_c', Quantity(delta_c))
        ])

    @staticmethod
    def evaluate(lon, lat, r_s, delta_c):
        """Evaluate the model (static function)."""
        pass


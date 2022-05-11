# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Pulsar velocity distribution models."""
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.units import Quantity

__all__ = [
    "FaucherKaspi2006VelocityBimodal",
    "FaucherKaspi2006VelocityMaxwellian",
    "Paczynski1990Velocity",
    "velocity_distributions",
]

# Simulation range used for random number drawing
VMIN, VMAX = Quantity([0, 4000], "km/s")


class FaucherKaspi2006VelocityMaxwellian(Fittable1DModel):
    r"""Maxwellian pulsar velocity distribution.

    .. math::
        f(v) = A \sqrt{ \frac{2}{\pi}} \frac{v ^ 2}{\sigma ^ 3 }
               \exp \left(-\frac{v ^ 2}{2 \sigma ^ 2} \right)

    Reference: https://ui.adsabs.harvard.edu/abs/2006ApJ...643..332F

    Parameters
    ----------
    amplitude : float
        Value of the integral
    sigma : float
        Velocity parameter (km s^-1)
    """

    amplitude = Parameter()
    sigma = Parameter()

    def __init__(self, amplitude=1, sigma=265, **kwargs):
        super().__init__(amplitude=amplitude, sigma=sigma, **kwargs)

    @staticmethod
    def evaluate(v, amplitude, sigma):
        """One dimensional velocity model function."""
        term1 = np.sqrt(2 / np.pi) * v**2 / sigma**3
        term2 = np.exp(-(v**2) / (2 * sigma**2))
        return term1 * term2


class FaucherKaspi2006VelocityBimodal(Fittable1DModel):
    r"""Bimodal pulsar velocity distribution - Faucher & Kaspi (2006).

    .. math::
        f(v) = A\sqrt{\frac{2}{\pi}} v^2 \left[\frac{w}{\sigma_1^3}
        \exp \left(-\frac{v^2}{2\sigma_1^2} \right) + \frac{1-w}{\sigma_2^3}
        \exp \left(-\frac{v^2}{2\sigma_2^2} \right) \right]

    Reference: https://ui.adsabs.harvard.edu/abs/2006ApJ...643..332F (Formula (7))

    Parameters
    ----------
    amplitude : float
        Value of the integral
    sigma1 : float
        See model formula
    sigma2 : float
        See model formula
    w : float
        See model formula
    """

    amplitude = Parameter()
    sigma_1 = Parameter()
    sigma_2 = Parameter()
    w = Parameter()

    def __init__(self, amplitude=1, sigma_1=160, sigma_2=780, w=0.9, **kwargs):
        super().__init__(
            amplitude=amplitude, sigma_1=sigma_1, sigma_2=sigma_2, w=w, **kwargs
        )

    @staticmethod
    def evaluate(v, amplitude, sigma_1, sigma_2, w):
        """One dimensional Faucher-Guigere & Kaspi 2006 velocity model function."""
        A = amplitude * np.sqrt(2 / np.pi) * v**2
        term1 = (w / sigma_1**3) * np.exp(-(v**2) / (2 * sigma_1**2))
        term2 = (1 - w) / sigma_2**3 * np.exp(-(v**2) / (2 * sigma_2**2))
        return A * (term1 + term2)


class Paczynski1990Velocity(Fittable1DModel):
    r"""Distribution by Lyne 1982 and adopted by Paczynski and Faucher.

    .. math::
        f(v) = A\frac{4}{\pi} \frac{1}{v_0 \left[1 + (v / v_0) ^ 2 \right] ^ 2}

    Reference: https://ui.adsabs.harvard.edu/abs/1990ApJ...348..485P (Formula (3))

    Parameters
    ----------
    amplitude : float
        Value of the integral
    v_0 : float
        Velocity parameter (km s^-1)
    """

    amplitude = Parameter()
    v_0 = Parameter()

    def __init__(self, amplitude=1, v_0=560, **kwargs):
        super().__init__(amplitude=amplitude, v_0=v_0, **kwargs)

    @staticmethod
    def evaluate(v, amplitude, v_0):
        """One dimensional Paczynski 1990 velocity model function."""
        return amplitude * 4.0 / (np.pi * v_0 * (1 + (v / v_0) ** 2) ** 2)


"""Velocity distributions (dict mapping names to classes)."""
velocity_distributions = {
    "H05": FaucherKaspi2006VelocityMaxwellian,
    "F06B": FaucherKaspi2006VelocityBimodal,
    "F06P": Paczynski1990Velocity,
}

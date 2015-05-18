# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from sherpa.models import ArithmeticModel, Parameter


class MyPLExpCutoff(ArithmeticModel):
    """
    Model definition for a power law with exponential cutoff
    Rescaled all parameters to TeV.
    """

    def __init__(self, name='myplexpcutoff'):
        self.Eo = Parameter(name, 'Eo', 1, frozen=True, units='keV')  # p[0] Normalized at 1 TeV by default
        self.beta = Parameter(name, 'beta', 1e-1, min=1e-3, max=10, units='1/TeV')  # p[1]
        self.gamma = Parameter(name, 'gamma', 2, min=-1, max=5)  # p[2]
        self.No = Parameter(name, 'No', 1e-11, min=1e-15, max=1e-5, units='1/cm^2/s/TeV')  # p[3]

        ArithmeticModel.__init__(self, name, (self.Eo, self.beta, self.gamma, self.No))

    def plec(self, E, p):
        return 1e-9 * p[3] * np.exp(-E * p[1] * 1e-9) * (E / p[0] / 1e9) ** -p[2]

    def point(self, p, x):
        """
         point version, dN/dE = No * (E/Eo)^-gamma * exp(-beta*E) ### (beta = 1/Ecut)

        Params
        `p`  list of ordered parameter values.
        `x`  ndarray of bin midpoints.

        returns ndarray of calculated function values at
                bin midpoints
        """
        return self.plec(x, p)

    def integrated(self, p, xlo, xhi):
        """
         integrated form from lower bin edge to upper edge

        Params
        `p`   list of ordered parameter values.
        `xlo` ndarray of lower bin edges.
        `xhi` ndarray of upper bin edges.

        returns ndarray of integrated function values over
                lower and upper bin edges.
        """
        flux = (xhi - xlo) * (self.plec(xhi, p) + self.plec(xlo, p)) / 2  # Trapezoid integration
        return flux

    def calc(self, p, xlo, xhi=None, *args, **kwargs):
        """
        Params
        `p`   list of ordered parameter values.
        `x`   ndarray of domain values, bin midpoints or lower
              bin edges.
        `xhi` ndarray of upper bin edges.

        returns ndarray of calculated function values.
        """
        if xhi is None:
            return self.point(p, xlo)
        return self.integrated(p, xlo, xhi)

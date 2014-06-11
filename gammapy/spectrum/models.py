# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Common spectral models used in gamma-ray astronomy.

A Model can be either a TableModel, which is represented
as arrays of energies and fluxes. Or a AnalyticModel, which
is represented by an analytic function flux(energy).
"""
from __future__ import print_function, division
import numpy as np
from numpy import pi, exp, log, log10
from astropy import constants as const
from astropy.units import Unit

__all__ = ['AnalyticModel', 'BlackBody', 'BrokenPowerLaw', 'CompositeModel',
           'LogParabola', 'Model', 'PLExpCutoff', 'PowerLaw', 'TableModel']

# Define some constatns
MeV_to_GeV = Unit('MeV').to(Unit('GeV'))
MeV_to_erg = Unit('MeV').to(Unit('erg'))
erg_to_eV = Unit('erg').to(Unit('eV'))
c = const.c.cgs.value
hbar = const.hbar.cgs.value
k_B = const.k_B.cgs.value
k_B_eV = const.k_B.to('eV/K').value


class Model(object):
    """Abstract base class for all spectra."""

    def __call__(self, e, power=0):
        """Make spectrum have value 0 outside range emin < e < emax"""
        mask = (e == e)
        if self.emin:
            mask &= (e > self.emin)
        if self.emax:
            mask &= (e < self.emax)
        return np.where(mask, e ** power * self._y(e), 0)

    def __add__(self, other):
        """Spectra can be added: s = s1 + s2"""
        return CompositeModel(self, other, 'add')

    def __pow__(self, other):
        """Spectra can be taken to a power: s = s1 ** s2.
        Really this will only be used for numbers s2."""
        return CompositeModel(self, other, 'pow')

    def _elim(self, emin=None, emax=None):
        if emin == None:
            emin = self.emin
        if emax == None:
            emax = self.emax
        # @todo: Is this necessary?
        # emin = np.asarray(emin, dtype=float)
        # emax = np.asarray(emax, dtype=float)
        return np.array((emin, emax))

    def points(self, emin=None, emax=None, power=0, npoints=100):
        emin, emax = self._elim(emin, emax)
        e = np.logspace(log10(emin), log10(emax), npoints)
        y = self.__call__(e, power)
        return e, y

    def table_spectrum(self, emin=None, emax=None, npoints=100):
        """Make a table spectrum"""
        e, y = self.points(emin, emax, npoints)
        return TableModel(e, y, emin, emax)

    def _log_integral(self, emin=None, emax=None, power=0):
        """Integrate over x = log(e) instead of e for numerical precision and speed.
        Try to remember your high-school calculus and transform the integral.
        """
        from scipy.integrate import quad
        f = (lambda x: self(x, power) * x)
        return quad(f, log(emin), log(emax))

    def _lin_integral(self, emin=None, emax=None, power=0):
        """Integrate over e directly, not going to log energy scale"""
        from scipy.integrate import quad
        return quad(self, emin, emax, power)

    def integral(self, emin=None, emax=None, power=0, method='lin'):
        """Compute integral"""
        emin, emax = self._elim(emin, emax)
        integrators = {'log': self._log_integral,
                       'lin': self._lin_integral}
        try:
            return integrators[method](emin, emax, power)
        except ValueError:
            # quad can't handle arrays, so we do the for loop ourselves.
            # @todo: I'm not sure what the best way to detect that case is here?
            result = np.empty_like(emin)
            for ii in np.arange(len(result)):
                result[ii] = integrators[method](emin[ii], emax[ii], power)[0]
            return result


class CompositeModel(Model):
    """Model that is the binary composition of two other spectra."""
    def __init__(self, spec1, spec2, op):
        self.spec1 = spec1
        self.spec2 = spec2
        self.op = op

    def __call__(self, x, power=0):
        if self.op == 'add':
            return self.spec1(x, power) + self.spec2(x, power)
        elif self.op == 'pow':
            return self.spec1(x, power) ** self.spec2(x, power)
        else:
            raise NotImplementedError


class TableModel(Model):
    """A spectrum represented by numeric arrays of x and y values.

    Internally all calculations are done on log10(x) and log10(y)
    for numerical stability and accuracy.
    """
    def __init__(self, e, y, emin=None, emax=None):
        from scipy.interpolate import interp1d
        self.emin = emin if emin else e.min()
        self.emax = emax if emax else e.max()
        self.e = np.asarray(e)
        # self.de = np.ediff1d(e, to_end=e[-1]-e[-2])
        self.y = np.ones_like(e) * np.asarray(y)
        self._y = interp1d(self.e, self.y, fill_value=0,
                           bounds_error=False, kind='cubic')

    @staticmethod
    def zeros(emin, emax, npoints=100):
        """Factory function to create a TableModel
        with all zeros"""
        e = np.logspace(log10(emin), log10(emax), npoints)
        return TableModel(e, 0, emin, emax)

    def get_points(self):
        return self.x, self.y

    def integral(self, emin=None, emax=None, power=0):
        emin, emax = self._elim(emin, emax)
        mask = np.where(emin < self.x < emax)
        return (self.x ** power * self.y * self.dx)[mask].sum()

    def plot(self, power=2):
        import matplotlib.pyplot as plt
        plt.plot(self.x, self.x ** power * self.y)


class AnalyticModel(Model):
    """Spectrum represented by an analytic function."""

    def __str__(self):
        s = '{0}\n'.format(self.__class__.__name__)
        for par in self.pars:
            if par.vary:
                err = '+/- {0:.3e}'.format(par.stderr)
            else:
                err = ' (fixed)'
            fmt = '{0:20s} = {1:.3e} {2}\n'
            s += (fmt.format(par.name, par.value, err))
        return s

    def error(self, E):
        return self._y_with_error(E)

    def plot(self, label, xlim, ylim, npoints=100,
             xscale=MeV_to_GeV, xpower=2, yscale=MeV_to_erg):
        import matplotlib.pyplot as plt
        x = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), npoints)
        y = self(x)
        # Scale
        y *= yscale * x ** xpower
        x *= xscale
        plt.plot(x, y, label=label)


class PowerLaw(AnalyticModel):
    """Power law model."""

    def __init__(self, Flux_Density, Spectral_Index,
                 Pivot_Energy, emin=None, emax=None):
        self.pars = [Flux_Density, Spectral_Index, Pivot_Energy]
        self.emin = emin
        self.emax = emax

    @staticmethod
    def _formula(E, N0, g, E0):
        return N0 * (E / E0) ** g

    def _y(self, E):
        N0, g, E0 = [par.value for par in self.pars]
        return PowerLaw._formula(E, N0, g, E0)

    def _y_with_error(self, E):
        from uncertainties import ufloat
        N0, g, E0 = [ufloat(par.value, par.stderr, par.name)
                     for par in self.pars]
        return PowerLaw._formula(E, N0, g, E0)

    def integral(self, emin=None, emax=None, power=0):
        """Compute integral using analytic formula (fast and exact)"""
        # @todo: should be implemented in log for speed and stability,
        # but how???
        emin, emax = self._elim(emin, emax)
        N0, g, E0 = [par.value for par in self.pars]
        g1 = g + power + 1
        E0 = np.asarray(E0, dtype=float)
        n = (N0 * E0) / g1
        upper = (emax / E0) ** g1
        lower = (emin / E0) ** g1
        return n * (upper - lower)

    @staticmethod
    def factory(Flux_Density=1, Spectral_Index=2, Pivot_Energy=1):
        from lmfit import Parameter
        pars = [Parameter(name='Flux_Density', value=Flux_Density),
                Parameter(name='Spectral_Index', value=Spectral_Index),
                Parameter(name='Pivot_Energy', value=Pivot_Energy)]
        return PowerLaw(*pars)


class PLExpCutoff(AnalyticModel):
    """Power law model with exponential cutoff."""

    def __init__(self, Flux_Density, Spectral_Index,
                 Cutoff, Pivot_Energy, emin=None, emax=None):
        self.pars = [Flux_Density, Spectral_Index, Cutoff, Pivot_Energy]
        self.emin = emin
        self.emax = emax

    @staticmethod
    def _formula(E, N0, g, Ecut, E0):
        return N0 * (E / E0) ** g * exp(-E / Ecut)

    def _y(self, E):
        N0, g, Ecut, E0 = [par.value for par in self.pars]
        return PLExpCutoff._formula(E, N0, g, Ecut, E0)

    def _y_with_error(self, E):
        from uncertainties import ufloat
        N0, g, Ecut, E0 = [ufloat(par.value, par.stderr, par.name)
                           for par in self.pars]
        return PLExpCutoff._formula(E, N0, g, Ecut, E0)


class LogParabola(AnalyticModel):
    """Log parabola model."""

    def __init__(self, Flux_Density, Spectral_Index,
                 beta, Pivot_Energy, emin=None, emax=None):
        self.pars = [Flux_Density, Spectral_Index, beta, Pivot_Energy]
        self.emin = emin
        self.emax = emax

    @staticmethod
    def _formula(E, N0, g, beta, E0):
        power = -(g + beta * log(E / E0))
        return N0 * (E / E0) ** power

    def _y(self, E):
        N0, g, beta, E0 = [par.value for par in self.pars]
        return LogParabola._formula(E, N0, g, beta, E0)

    def _y_with_error(self, E):
        from uncertainties import ufloat
        N0, g, beta, E0 = [ufloat(par.value, par.stderr, par.name)
                           for par in self.pars]
        return LogParabola._formula(E, N0, g, beta, E0)


class BrokenPowerLaw(AnalyticModel):
    """Broken power-law model."""

    def __init__(self, Flux_Density, Spectral_Index, Spectral_Index2,
                 Break_Energy, emin=None, emax=None):
        self.pars = [Flux_Density, Spectral_Index, Spectral_Index2,
                     Break_Energy]
        self.emin = emin
        self.emax = emax

    @staticmethod
    def _formula(E, N0, g, g2, Eb):
        # @todo: I don't like the fact that Eb is used as E0!!!
        # Probably best to have an independent E0 in the formula?
        return np.where(E < Eb,
                        N0 * (E / Eb) ** g,
                        N0 * (E / Eb) ** g2)

    def _y(self, E):
        N0, g, g2, Eb = [par.value for par in self.pars]
        return BrokenPowerLaw._formula(E, N0, g, g2, Eb)

    def _y_with_error(self, E):
        from uncertainties import ufloat
        N0, g, g2, Eb = [ufloat(par.value, par.stderr, par.name)
                         for par in self.pars]
        return LogParabola._formula(E, N0, g, g2, Eb)

'''
@uncertainties.wrap
def I(e1, e2, e0, f0, g, ec = numpy.nan):
    """
    Compute integral flux for an exponentially cutoff power law.

    e1, e2  = energy integration range
    e0 = pivot energy
    f0 = flux density (at e0 if no cutoff is present)
    g  = spectral index
    ec = cutoff energy (optional, no cutoff if numpy.nan is given)
    """
    return quad(lambda e: ecpl_f(e, e0, f0, g, ec), e1, e2)[0]
'''


class BlackBody(AnalyticModel):
    """Black-body model.

    The energy density can be specified independently of the temperature.
    This is sometimes called a "gray body" spectrum.

    Photon number density (cm^-3 eV^-1)

    Parameters
    ----------
    T : float
        Temperature (K)
    W : float
        Energy density (eV cm^-3)
    """
    def __init__(self, T=2.7, W=None, emin=None, emax=None):
        self.emin = emin
        self.emax = emax
        self.T = T
        self.W = W if W else self._omega_b(T)

    def _omega_b(self, T):
        """Blackbody energy density.

        Parameters
        ----------
        T : array_like
            Temperature (K).

        Returns
        -------
        omega : `numpy.array`
            Blackbody energy density (eV cm^-3).
        """
        return (erg_to_eV *
                (pi ** 2 * (k_B * T) ** 4) /
                (15 * (hbar * c) ** 3))

    def _omega_g_over_b(self):
        """Returns: Fraction of blackbody energy density"""
        return self.W / self._omega_b(self.T)

    def _y(self, E):
        """Evaluate model.

        Parameters
        ----------
        E : array_like
            Energy (eV)
        """
        # Convert (k_B * T) to eV
        return ((15 * self.W / (np.pi ** 4 * (k_B_eV * self.T) ** 4) *
                E ** 2 / (exp(E / (k_B_eV * self.T)) - 1)))

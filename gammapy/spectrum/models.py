# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from ..extern.bunch import Bunch
from ..utils.energy import EnergyBounds
from .utils import integrate_spectrum

# This cannot be made a delayed import because the pytest matrix fails if it is
# https://travis-ci.org/gammapy/gammapy/jobs/151539845#L1799
try:
    from .sherpa_models import SherpaExponentialCutoffPowerLaw
except ImportError:
    pass


__all__ = [
    'SpectralModel',
    'PowerLaw',
    'PowerLaw2',
    'ExponentialCutoffPowerLaw',
    'ExponentialCutoffPowerLaw3FGL',
    'LogParabola',
    'TableModel',
]


class SpectralModel(object):
    """Spectral model base class.

    Derived classes should store their parameters as ``Bunch`` in an instance
    attribute called ``parameters``, see for example
    `~gammapy.spectrum.models.PowerLaw`.
    """

    def __call__(self, energy):
        """Call evaluate method of derived classes"""
        return self.evaluate(energy, **self.parameters)

    def __str__(self):
        """String representation"""
        ss = self.__class__.__name__
        for parname, parval in self.parameters.items():
            ss += '\n{parname} : {parval:.3g}'.format(**locals())
        return ss

    def integral(self, emin, emax, **kwargs):
        """
        Integrate spectral model numerically.

        .. math::

            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE

        kwargs are forwared to :func:`~gammapy.spectrum.integrate_spectrum``.

        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range
        """
        return integrate_spectrum(self, emin, emax, **kwargs)

    def energy_flux(self, emin, emax, **kwargs):
        """
        Compute energy flux in given energy range.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE

        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range
        """

        def f(x):
            return x * self(x)

        return integrate_spectrum(f, emin, emax, **kwargs)

    def to_dict(self):
        """Serialize to dict"""
        retval = dict()

        retval['name'] = self.__class__.__name__
        retval['parameters'] = list()
        for parname, parval in self.parameters.items():
            retval['parameters'].append(dict(name=parname,
                                             val=parval.value,
                                             unit=str(parval.unit)))
        return retval

    @classmethod
    def from_dict(cls, val):
        """Serialize from dict"""
        classname = val['name']
        kwargs = dict()
        for _ in val['parameters']:
            kwargs[_['name']] = _['val'] * u.Unit(_['unit'])
        return globals()[classname](**kwargs)

    def plot(self, energy_range, ax=None,
             energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
             energy_power=0, n_points=100, **kwargs):
        """Plot `~gammapy.spectrum.SpectralModel`

        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_range : `~astropy.units.Quantity`
            Plot range
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int, optional
            Power of energy to multiply flux axis with
        n_points : int, optional
            Number of evaluation nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        emin, emax = energy_range
        energy = EnergyBounds.equal_log_spacing(
            emin, emax, n_points, energy_unit)

        # evaluate model
        flux = self(energy).to(flux_unit)

        eunit = [_ for _ in flux.unit.bases if _.physical_type == 'energy'][0]

        y = (flux * np.power(energy, energy_power)
             ).to(flux.unit * eunit ** energy_power)

        ax.plot(energy.value, y.value, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy.unit))
        if energy_power > 0:
            ax.set_ylabel('E{0} * Flux [{1}]'.format(energy_power, y.unit))
        else:
            ax.set_ylabel('Flux [{}]'.format(y.unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        return ax

    def to_sherpa(self, name='default'):
        """Convert to Sherpa model

        To be implemented by subclasses
        """
        raise NotImplementedError('{}'.format(self.__class__.__name__))

    def spectral_index(self, energy, epsilon=1E-5):
        """
        Compute spectral index at given energy using a local powerlaw
        approximation.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which to estimate the index
        epsilon : float
            Fractional energy increment to use for determining the spectral index.

        Returns
        -------
        index : float
            Estimated spectral index.
        """
        f1 = self(energy)
        f2 = self(energy * (1 + epsilon))
        return np.log(f1 / f2) / np.log(1 + epsilon)


class PowerLaw(SpectralModel):
    r"""Spectral power-law model.

    .. math::

        \phi(E) = \phi_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : float, `~astropy.units.Quantity`
        :math:`Phi_0`
    reference : float, `~astropy.units.Quantity`
        :math:`E_0`
    """

    def __init__(self, index, amplitude, reference):
        self.parameters = Bunch(index=index,
                                amplitude=amplitude,
                                reference=reference)

    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        return amplitude * (energy / reference) ** (-index)

    def integral(self, emin, emax):
        r"""
        Integrate power law analytically.

        .. math::

            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE = \left.
            \phi_0 \frac{E_0}{-\Gamma + 1} \left( \frac{E}{E_0} \right)^{-\Gamma + 1}
            \right \vert _{E_{min}}^{E_{max}}


        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range.

        """
        pars = self.parameters

        val = -1 * pars.index + 1
        prefactor = pars.amplitude * pars.reference / val
        upper = (emax / pars.reference) ** val
        lower = (emin / pars.reference) ** val
        return prefactor * (upper - lower)

    def energy_flux(self, emin, emax):
        r"""
        Compute energy flux in given energy range analytically.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE = \left.
            \phi_0 \frac{E_0^2}{-\Gamma + 2} \left( \frac{E}{E_0} \right)^{-\Gamma + 2}
            \right \vert _{E_{min}}^{E_{max}}


        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range
        """
        pars = self.parameters
        val = -1 * pars.index + 2

        try:
            val_zero = (val.n == 0)
        except AttributeError:
            val_zero = (val == 0)

        if val_zero:
            # see https://www.wolframalpha.com/input/?i=a+*+x+*+(x%2Fb)+%5E+(-2)
            # for reference
            return pars.amplitude * pars.reference ** 2 * np.log(emax / emin)
        else:
            prefactor = pars.amplitude * pars.reference ** 2 / val
            upper = (emax / pars.reference) ** val
            lower = (emin / pars.reference) ** val
            return prefactor * (upper - lower)


    def to_sherpa(self, name='default'):
        """Return Sherpa `~sherpa.models.PowLaw1d`

        Parameters
        ----------
        name : str, optional
            Name of the sherpa model instance
        """
        import sherpa.models as m
        model = m.PowLaw1D('powlaw1d.' + name)
        model.gamma = self.parameters.index.value
        model.ref = self.parameters.reference.to('keV').value
        model.ampl = self.parameters.amplitude.to('cm-2 s-1 keV-1').value
        return model


class PowerLaw2(SpectralModel):
    r"""
    Spectral power-law model with integral as norm parameter

    See http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
    for further details.

    .. math::

        \phi(E) = F_0 \cdot \frac{\Gamma + 1}{E_{0, max}^{\Gamma + 1}
         - E_{0, min}^{\Gamma + 1}} \cdot E^{-\Gamma}

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        Spectral index :math:`\Gamma`
    amplitude : float, `~astropy.units.Quantity`
        Integral flux :math:`F_0`.
    emin : float, `~astropy.units.Quantity`
        Lower energy limit :math:`E_{0, min}`.
    emax : float, `~astropy.units.Quantity`
        Upper energy limit :math:`E_{0, max}`.

    """

    def __init__(self, amplitude, index, emin, emax):
        self.parameters = Bunch(amplitude=amplitude,
                                index=index,
                                emin=emin,
                                emax=emax)

    @staticmethod
    def evaluate(energy, amplitude, index, emin, emax):
        top = -index + 1
        bottom = emax ** (-index + 1) - emin ** (-index + 1)
        return amplitude * (top / bottom) * np.power(energy, -index)

    def integral(self, emin, emax):
        r"""
        Integrate power law analytically.

        .. math::

            F(E_{min}, E_{max}) = F_0 \cdot \frac{E_{max}^{\Gamma + 1} \
                                - E_{min}^{\Gamma + 1}}{E_{0, max}^{\Gamma + 1} \
                                - E_{0, min}^{\Gamma + 1}}

        Parameters
        ----------
        emin : float, `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : float, `~astropy.units.Quantity`
            Upper bound of integration range

        """
        pars = self.parameters
        top = np.power(emax, -pars.index + 1) - np.power(emin, -pars.index + 1)
        bottom = np.power(pars.emax, -pars.index + 1) - \
            np.power(pars.emin, -pars.index + 1)

        return pars.amplitude * top / bottom


class ExponentialCutoffPowerLaw(SpectralModel):
    r"""Spectral exponential cutoff power-law model.

    .. math::

        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma} \exp(-\lambda E)

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : float, `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : float, `~astropy.units.Quantity`
        :math:`E_0`
    lambda : float, `~astropy.units.Quantity`
        :math:`\lambda`
    """

    def __init__(self, index, amplitude, reference, lambda_):
        self.parameters = Bunch(index=index,
                                amplitude=amplitude,
                                reference=reference,
                                lambda_=lambda_)

    @staticmethod
    def evaluate(energy, index, amplitude, reference, lambda_):
        pwl = amplitude * (energy / reference) ** (-index)
        try:
            cutoff = np.exp(-energy * lambda_)
        except AttributeError:
            from uncertainties.unumpy import exp
            cutoff = exp(-energy * lambda_)
        return pwl * cutoff

    def to_sherpa(self, name='default'):
        """Return Sherpa `~sherpa.models.Arithmetic model`

        Parameters
        ----------
        name : str, optional
            Name of the sherpa model instance
        """
        model = SherpaExponentialCutoffPowerLaw(name='ecpl.' + name)
        pars = self.parameters
        model.gamma = pars.index.value
        model.ref = pars.reference.to('keV').value
        model.ampl = pars.amplitude.to('cm-2 s-1 keV-1').value
        # Sherpa ExponentialCutoffPowerLaw expects cutoff in 1/TeV
        model.cutoff = pars.lambda_.to('TeV-1').value

        return model


class ExponentialCutoffPowerLaw3FGL(SpectralModel):
    r"""Spectral exponential cutoff power-law model used for 3FGL.

    Note that the parmatrization is different from `ExponentialCutoffPowerLaw`:

    .. math::

        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma}
                  \exp \left( \frac{E_0 - E}{E_{C}} \right)

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : float, `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : float, `~astropy.units.Quantity`
        :math:`E_0`
    ecut : float, `~astropy.units.Quantity`
        :math:`E_{C}`
    """

    def __init__(self, index, amplitude, reference, ecut):
        self.parameters = Bunch(index=index,
                                amplitude=amplitude,
                                reference=reference,
                                ecut=ecut)

    @staticmethod
    def evaluate(energy, index, amplitude, reference, ecut):
        pwl = amplitude * (energy / reference) ** (-index)
        try:
            cutoff = np.exp((reference - energy) / ecut)
        except AttributeError:
            from uncertainties.unumpy import exp
            cutoff = exp((reference - energy) / ecut)
        return pwl * cutoff


class LogParabola(SpectralModel):
    r"""Spectral log parabola model.

    .. math::

        f(x) = A \left( \frac{E}{E_0} \right) ^ {
          - \alpha - \beta \log{ \left( \frac{E}{E_0} \right) }
        }

    Parameters
    ----------
    amplitude : float, `~astropy.units.Quantity`
        :math:`Phi_0`
    reference : float, `~astropy.units.Quantity`
        :math:`E_0`
    alpha : float, `~astropy.units.Quantity`
        :math:`\alpha`
    beta : float, `~astropy.units.Quantity`
        :math:`\beta`
    """

    def __init__(self, amplitude, reference, alpha, beta):
        self.parameters = Bunch(amplitude=amplitude,
                                reference=reference,
                                alpha=alpha,
                                beta=beta)

    @staticmethod
    def evaluate(energy, amplitude, reference, alpha, beta):
        # cast dimensionless values as np.array, because of bug in Astropy < v1.2
        # https://github.com/astropy/astropy/issues/4764
        try:
            xx = (energy / reference).to('')
            exponent = -alpha - beta * np.log(xx)
        except AttributeError:
            from uncertainties.unumpy import log
            xx = energy / reference
            exponent = -alpha - beta * log(xx)
        return amplitude * np.power(xx, exponent)


class TableModel(SpectralModel):
    """A model generated from a table of energy and value arrays.

    The units returned will be the units of the values array provided at
    initialization. The model will return values interpolated in
    log-space, returning 0 for energies outside of the limits of the provided
    energy array.

    Class implementation follows closely what has been done in
    `naima.models.TableModel`

    Parameters
    ----------
    energy : `~astropy.units.Quantity` array
        Array of energies at which the model values are given
    values : array
        Array with the values of the model at energies ``energy``.
    amplitude : float
        Model amplitude that is multiplied to the supplied arrays. Defaults to 1.
    """

    def __init__(self, energy, values, amplitude=1):
        from scipy.interpolate import interp1d
        self.parameters = Bunch(amplitude=amplitude)
        self.energy = energy
        self.values = values

        loge = np.log10(self.energy.to('eV').value)
        try:
            self.unit = self.values.unit
            logy = np.log10(self.values.value)
        except AttributeError:
            self.unit = u.Unit('')
            logy = np.log10(self.values)

        self.interplogy = interp1d(loge,
                                   logy,
                                   fill_value=-np.Inf,
                                   bounds_error=False,
                                   kind='cubic')

    def evaluate(self, energy, amplitude):
        interpy = np.power(10, self.interplogy(
            np.log10(energy.to('eV').value))
        )
        return amplitude * interpy * self.unit

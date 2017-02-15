# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
import astropy.units as u
from astropy.table import Table

from ..extern.bunch import Bunch
from ..utils.energy import EnergyBounds
from .utils import integrate_spectrum
from ..utils.scripts import make_path
from ..utils.modeling import Parameter, ParameterList

# This cannot be made a delayed import because the pytest matrix fails if it is
# https://travis-ci.org/gammapy/gammapy/jobs/151539845#L1799
try:
    from .sherpa_utils import SherpaExponentialCutoffPowerLaw
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
    'AbsorbedSpectralModel',
]


class SpectralModel(object):
    """Spectral model base class.

    Derived classes should store their parameters as
    `~gammapy.spectrum.models.ParameterList`, see for example return pardict
    `~gammapy.spectrum.models.PowerLaw`.
    """
    def __call__(self, energy):
        """Call evaluate method of derived classes"""
        kwargs = dict()
        for par in self.parameters.parameters:
            kwargs[par.name] = par.quantity
        return self.evaluate(energy, **kwargs)

    def __str__(self):
        """String representation"""
        ss = self.__class__.__name__
        ss += '\n{}'.format(self.parameters)
        return ss

    def _parse_uarray(self, uarray):
        from uncertainties import unumpy
        values = unumpy.nominal_values(uarray)
        errors = unumpy.std_devs(uarray)
        return values, errors

    def evaluate_error(self, energy):
        """
        Evaluate spectral model with error propagation.

        Parameters
        ----------
        energy : `~astropy.units.quantity`
            Energy at which to evaluate.

        Returns
        -------
        flux, flux_error : tuple of `~astropy.units.Quantity`
            Tuple of flux and flux error.
        """
        unit = self(energy).unit
        upars = self.parameters._ufloats
        uarray = self.evaluate(energy.value, **upars)
        return self._parse_uarray(uarray) * unit

    def integral(self, emin, emax, **kwargs):
        """
        Integrate spectral model numerically.

        .. math::

            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE

        kwargs are forwared to :func:`~gammapy.spectrum.integrate_spectrum`.

        If array input for ``emin`` and ``emax`` is given you have to set
        ``intervals=True`` if you want the integral in each energy bin.

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : `~astropy.units.Quantity`
            Upper bound of integration range
        """
        return integrate_spectrum(self, emin, emax, **kwargs)

    def integral_error(self, emin, emax, **kwargs):
        """
        Integrate spectral model numerically with error propagation.

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : `~astropy.units.Quantity`
            Upper bound of integration range

        Returns
        -------
        integral, integral_error : tuple of `~astropy.units.Quantity`
            Tuple of integral flux and integral flux error.
        """
        unit = self.integral(emin, emax, **kwargs).unit
        upars = self.parameters._ufloats

        def f(x):
            return self.evaluate(x, **upars)

        uarray = integrate_spectrum(f, emin.value, emax.value, **kwargs)
        return self._parse_uarray(uarray) * unit

    def energy_flux(self, emin, emax, **kwargs):
        """
        Compute energy flux in given energy range.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : `~astropy.units.Quantity`
            Upper bound of integration range
        """

        def f(x):
            return x * self(x)
        return integrate_spectrum(f, emin, emax, **kwargs)

    def energy_flux_error(self, emin, emax, **kwargs):
        """
        Compute energy flux in given energy range with error propagation.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : `~astropy.units.Quantity`
            Upper bound of integration range

        Returns
        -------
        energy_flux, energy_flux_error : tuple of `~astropy.units.Quantity`
            Tuple of energy flux and energy flux error.

        """

        unit = self.energy_flux(emin, emax, **kwargs).unit
        upars = self.parameters._ufloats

        def f(x):
            return x * self.evaluate(x, **upars)

        uarray = integrate_spectrum(f, emin.value, emax.value, **kwargs)
        return self._parse_uarray(uarray) * unit

    def to_dict(self):
        """Serialize to dict"""
        retval = dict()

        retval['name'] = self.__class__.__name__
        retval['parameters'] = list()
        for par in self.parameters.parameters:
            retval['parameters'].append(dict(name=par.name,
                                             val=float(par.value),
                                             unit=str(par.unit)))
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

        y = self._plot_scale_flux(energy, flux, energy_power)

        ax.plot(energy.value, y.value, **kwargs)
        self._plot_format_ax(ax, energy, y, energy_power)
        return ax

    def plot_error(self, energy_range, ax=None,
             energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
             energy_power=0, n_points=100, **kwargs):
        """Plot error `~gammapy.spectrum.SpectralModel`

        kwargs are forwarded to :func:`~matplotlib.pyplot.fill_between`

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

        kwargs.setdefault('facecolor', 'black')
        kwargs.setdefault('alpha', 0.2)
        kwargs.setdefault('linewidth', 0)

        emin, emax = energy_range
        energy = EnergyBounds.equal_log_spacing(
            emin, emax, n_points, energy_unit)

        flux, flux_err = self.evaluate_error(energy)

        y_lo = self._plot_scale_flux(energy, flux - flux_err, energy_power)
        y_hi = self._plot_scale_flux(energy, flux + flux_err, energy_power)

        where = (y_hi > 0) & (energy >= energy_range[0]) & (energy <= energy_range[1])
        ax.fill_between(energy.value, y_lo.value, y_hi.value, where=where, **kwargs)
        self._plot_format_ax(ax, energy, y_lo, energy_power)
        return ax

    def _plot_format_ax(self, ax, energy, y, energy_power):
        ax.set_xlabel('Energy [{}]'.format(energy.unit))
        if energy_power > 0:
            ax.set_ylabel('E{0} * Flux [{1}]'.format(energy_power, y.unit))
        else:
            ax.set_ylabel('Flux [{}]'.format(y.unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')

    def _plot_scale_flux(self, energy, flux, energy_power):
        eunit = [_ for _ in flux.unit.bases if _.physical_type == 'energy'][0]

        y = (flux * np.power(energy, energy_power)
             ).to(flux.unit * eunit ** energy_power)
        return y

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

    def inverse(self, value, emin=0.1 * u.TeV, emax=100 * u.TeV):
        """
        Return energy for a given function value of the spectral model.

        Uses numerical root finding algorithm.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        emin : `~astropy.units.Quantity`
            Lower bracket value in case solution is not unique.
        emax : `~astropy.units.Quantity`
            Upper bracket value in case solution is not unique.
        """
        from scipy.optimize import brentq

        energies = []
        for val in np.atleast_1d(value):
            def f(x):
                # scale by 1E12 to achieve better precision
                y = self(x * u.TeV).to(value.unit).value
                return 1E12 * (y - val.value)

            energy = brentq(f, emin.to('TeV').value, emax.to('TeV').value)
            energies.append(energy)
        return energies * u.TeV

    def copy(self):
        return copy.deepcopy(self)


class PowerLaw(SpectralModel):
    r"""Spectral power-law model.

    .. math::

        \phi(E) = \phi_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`Phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    """

    def __init__(self, index, amplitude, reference):
        self.parameters = ParameterList([
            Parameter('index', index, parmin=0),
            Parameter('amplitude', amplitude, parmin=0),
            Parameter('reference', reference, frozen=True)
        ])

    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        return amplitude * np.power((energy / reference), -index)

    def integral(self, emin, emax, **kwargs):
        r"""
        Integrate power law analytically.

        .. math::

            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE = \left.
            \phi_0 \frac{E_0}{-\Gamma + 1} \left( \frac{E}{E_0} \right)^{-\Gamma + 1}
            \right \vert _{E_{min}}^{E_{max}}


        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : `~astropy.units.Quantity`
            Upper bound of integration range.

        """
        # kwargs are passed to this function but not used
        # this is to get a consistent API with SpectralModel.integral()
        pars = self.parameters

        if np.isclose(pars['index'].value, 1):
            e_unit = emin.unit
            prefactor = pars['amplitude'].quantity * pars['reference'].quantity.to(e_unit)
            upper = np.log(emax.to(e_unit).value)
            lower = np.log(emin.value)
        else:
            val = -1 * pars['index'].value + 1
            prefactor = pars['amplitude'].quantity * pars['reference'].quantity / val
            upper = np.power((emax / pars['reference'].quantity), val)
            lower = np.power((emin / pars['reference'].quantity), val)

        integral = prefactor * (upper - lower)
        return integral

    def energy_flux(self, emin, emax):
        r"""
        Compute energy flux in given energy range analytically.

        .. math::


            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE = \left.
            \phi_0 \frac{E_0^2}{-\Gamma + 2} \left( \frac{E}{E_0} \right)^{-\Gamma + 2}
            \right \vert _{E_{min}}^{E_{max}}


        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : `~astropy.units.Quantity`
            Upper bound of integration range
        """
        pars = self.parameters
        val = -1 * pars['index'].value + 2

        try:
            val_zero = np.isclose(val.n, 0)
        except AttributeError:
            val_zero = np.isclose(val, 0)

        if val_zero:
            # see https://www.wolframalpha.com/input/?i=a+*+x+*+(x%2Fb)+%5E+(-2)
            # for reference
            temp = pars['amplitude'].quantity * pars['reference'].quantity ** 2
            return temp * np.log(emax / emin)
        else:
            prefactor = pars['amplitude'].quantity * pars['reference'].quantity ** 2 / val
            upper = (emax / pars['reference'].quantity) ** val
            lower = (emin / pars['reference'].quantity) ** val
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
        model.gamma = self.parameters['index'].value
        model.ref = self.parameters['reference'].quantity.to('keV').value
        model.ampl = self.parameters['amplitude'].quantity.to('cm-2 s-1 keV-1').value
        return model

    def inverse(self, value):
        """
        Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        p = self.parameters
        base = value / p['amplitude'].quantity
        return p['reference'].quantity * np.power(base, - 1. / p['index'].value)


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
    index : `~astropy.units.Quantity`
        Spectral index :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        Integral flux :math:`F_0`.
    emin : `~astropy.units.Quantity`
        Lower energy limit :math:`E_{0, min}`.
    emax : `~astropy.units.Quantity`
        Upper energy limit :math:`E_{0, max}`.

    """

    def __init__(self, amplitude, index, emin, emax):
        self.parameters = ParameterList([
            Parameter('amplitude', amplitude, parmin=0),
            Parameter('index', index, parmin=0),
            Parameter('emin', emin),
            Parameter('emax', emax)
        ])

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
        emin : `~astropy.units.Quantity`
            Lower bound of integration range.
        emax : `~astropy.units.Quantity`
            Upper bound of integration range

        """
        pars = self.parameters
        temp1 = np.power(emax, -pars['index'].value + 1)
        temp2 = np.power(emin, -pars['index'].value + 1)
        top = temp1 - temp2
        temp1 = np.power(pars['emax'].quantity, -pars['index'].value + 1)
        temp2 = np.power(pars['emin'].quantity, -pars['index'].value + 1)
        bottom = temp1 - temp2

        return pars['amplitude'].quantity * top / bottom

    def inverse(self, value):
        """
        Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        p = self.parameters
        index = p['index'].value
        top = -index + 1
        bottom = (p['emax'].quantity.to('TeV').value ** (-index + 1) -
                  p['emin'].quantity.to('TeV').value ** (-index + 1))
        term = (bottom / top) * (value / p['amplitude'].quantity).to('1 / TeV')
        return np.power(term.value, -1. / index) * u.TeV


class ExponentialCutoffPowerLaw(SpectralModel):
    r"""Spectral exponential cutoff power-law model.

    .. math::

        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma} \exp(-\lambda E)

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    lambda : `~astropy.units.Quantity`
        :math:`\lambda`
    """

    def __init__(self, index, amplitude, reference, lambda_):
        self.parameters = ParameterList([
            Parameter('index', index, parmin=0),
            Parameter('amplitude', amplitude, parmin=0),
            Parameter('reference', reference, frozen=True),
            Parameter('lambda_', lambda_, parmin=0)
        ])

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
        model.gamma = pars['index'].value
        model.ref = pars['reference'].quantity.to('keV').value
        model.ampl = pars['amplitude'].quantity.to('cm-2 s-1 keV-1').value
        # Sherpa ExponentialCutoffPowerLaw expects cutoff in 1/TeV
        model.cutoff = pars['lambda_'].quantity.to('TeV-1').value

        return model


class ExponentialCutoffPowerLaw3FGL(SpectralModel):
    r"""Spectral exponential cutoff power-law model used for 3FGL.

    Note that the parmatrization is different from `ExponentialCutoffPowerLaw`:

    .. math::

        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma}
                  \exp \left( \frac{E_0 - E}{E_{C}} \right)

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    ecut : `~astropy.units.Quantity`
        :math:`E_{C}`
    """

    def __init__(self, index, amplitude, reference, ecut):
        self.parameters = ParameterList([
            Parameter('index', index, parmin=0),
            Parameter('amplitude', amplitude, parmin=0),
            Parameter('reference', reference, frozen=0),
            Parameter('ecut', ecut)
        ])

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
    amplitude : `~astropy.units.Quantity`
        :math:`Phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    alpha : `~astropy.units.Quantity`
        :math:`\alpha`
    beta : `~astropy.units.Quantity`
        :math:`\beta`
    """

    def __init__(self, amplitude, reference, alpha, beta):
        self.parameters = ParameterList([
            Parameter('amplitude', amplitude, parmin=0),
            Parameter('reference', reference, frozen=True),
            Parameter('alpha', alpha),
            Parameter('beta', beta)
        ])

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
    scale_logy : boolean
        interpolation can be done linearly or in logarithm
    """

    def __init__(self, energy, values, amplitude=1, scale_logy=True):
        from scipy.interpolate import interp1d
        self.parameters = ParameterList([
            Parameter('amplitude', amplitude, parmin=0)
        ])
        self.energy = energy
        self.values = values
        self.scale_logy = scale_logy

        self.lo_threshold = energy[0]
        self.hi_threshold = energy[-1]

        loge = np.log10(self.energy.to('eV').value)
        try:
            self.unit = self.values.unit
            if scale_logy is True:
                y = np.log10(self.values.value)
            else:
                y = self.values.value
        except AttributeError:
            self.unit = u.Unit('')
            if scale_logy is True:
                y = np.log10(self.values)
            else:
                y = self.values
        self.interpy = interp1d(loge,
                                y,
                                fill_value=-np.Inf,
                                bounds_error=False,
                                kind='cubic')

    @classmethod
    def read_xspec_model(cls, filename, param):
        """A Table containing absorbed values from a XSPEC model
        as a function of energy.
        Todo:
        Format of the file should be described and discussed in
        https://gamma-astro-data-formats.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        filename : `str`
            File containing the XSPEC model
        param : float
            Model parameter value

        Examples
        --------
        Fill table from an EBL model (Franceschini, 2008)

        >>> from gammapy.spectrum.models import TableModel
        >>> filename = '$GAMMAPY_EXTRA/datasets/ebl/ebl_franceschini.fits.gz'
        >>> table_model = TableModel.read_xspec_model(filename=filename, param=0.3)
        """
        filename = str(make_path(filename))

        # Check if parameter value is in range
        table_param = Table.read(filename, hdu='PARAMETERS')
        param_min = table_param['MINIMUM']
        param_max = table_param['MAXIMUM']
        if param < param_min or param > param_max:
            err = 'Parameter out of range, param={0}, param_min={1}, param_max={2}'.format(
                param, param_min, param_max)
            raise ValueError(err)

        # Get energy values
        table_energy = Table.read(filename, hdu='ENERGIES')
        energy_lo = table_energy['ENERG_LO']
        energy_hi = table_energy['ENERG_HI']

        # Hack while format is not fixed, energy values are in keV
        energy_bounds = EnergyBounds.from_lower_and_upper_bounds(lower=energy_lo,
                                                                 upper=energy_hi,
                                                                 unit=u.keV)
        energy = energy_bounds.log_centers

        # Get spectrum values (no interpolation, take closest value for param)
        table_spectra = Table.read(filename, hdu='SPECTRA')
        idx = np.abs(table_spectra['PARAMVAL'] - param).argmin()
        values = table_spectra[idx][1] * u.Unit('')  # no dimension

        return cls(energy=energy, values=values, scale_logy=False)

    def evaluate(self, energy, amplitude):

        is_array = True
        try:
            len(energy)
        except:
            is_array = False

        # Not working for astropy.units.quantity.Quantity
        # if isinstance(energy, (np.ndarray, np.generic)):
        if is_array:  # Test if array
            # initialise array value to zero (dim energy)
            values = np.zeros(len(energy), dtype=float)
            # mask for energy range
            mask = (energy >= self.lo_threshold) & (
                energy <= self.hi_threshold)
            # apply interpolation for masked values
            values[mask] = self.interpy(np.log10(energy[mask].to('eV').value))
            # Get rid of negative values (due to interpolation)
            # Needed because of the rand.poisson used in SpectrumSimulation class
            # Should be fixed in the class itself ?
            if self.scale_logy is False:
                values[values < 0] = 0.
        else:  # if not array
            # test if energy is in range
            if (energy >= self.lo_threshold or energy <= self.hi_threshold):
                values = self.interpy(np.log10(energy.to('eV').value))
            else:
                values = 0

        if self.scale_logy:
                values = np.power(10, values)
        return amplitude * values * self.unit

    def plot(self, energy_range, ax=None, energy_unit='TeV',
             n_points=100, **kwargs):
        """Plot `~gammapy.spectrum.TableModel`

        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        energy_range : `~astropy.units.Quantity`
            Plot range
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
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

        y = self.interpy(
            np.log10(energy.to('eV').value)) * self.parameters['amplitude'].quantity
        if self.scale_logy:
            y = np.power(10, y)

        ax.plot(energy.value, y, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy.unit))

        ax.set_ylabel('Table model')
        ax.set_xscale("log", nonposx='clip')
        if self.scale_logy:
            ax.set_yscale("log", nonposy='clip')
        return ax


class AbsorbedSpectralModel(SpectralModel):

    def __init__(self, spectral_model, table_model):
        """Absorbed spectral model

        Parameters
        ----------
        spectral_model : `~gammapy.spectrum.models.SpectralModel`
            spectral model
        table_model : `~gammapy.spectrum.models.TableModel`
            table model
        """
        self.spectral_model = spectral_model
        self.table_model = table_model
        # Will be implemented later for sherpa fit
        self.parameters = ParameterList([])

    def evaluate(self, energy):
        flux = self.spectral_model.__call__(energy)
        absorption = self.table_model.__call__(energy)
        return flux * absorption

    def to_sherpa(self, name='default'):
        """Convert to Sherpa model

        To be implemented by subclasses
        """
        raise NotImplementedError('{}'.format(self.__class__.__name__))

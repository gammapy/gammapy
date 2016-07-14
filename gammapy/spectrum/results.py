# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.extern import six
from astropy.modeling import models
from astropy.table import Table, Column, QTable, hstack, vstack
import astropy.units as u

from ..spectrum import DifferentialFluxPoints, CountsSpectrum
from ..extern.bunch import Bunch
from ..utils.energy import EnergyBounds
from ..utils.scripts import read_yaml, make_path

__all__ = [
           'SpectrumFitResult',
           'SpectrumResult',
           ]

class SpectrumFitResult(object):
    """Class representing the result of a spectral fit

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Best-fit model
    covariance : array-like, optional
        Covariance matrix
    covar_axis : list, optional
        List of strings defining the parameter order in covariance
    fit_range : `~astropy.units.Quantity`
        Energy range of the spectral fit
    statname : str, optional
        Statistic used for the fit
    statval : float, optional
        Final fit statistic
    npred : array-like, optional
        On counts predicted by the fit
    flux_at_1TeV : dict, optional
        Flux for the fitted model at 1 TeV
    flux_at_1TeV_err : dict, optional
        Error on the flux for the fitted model at 1 TeV
    """
    def __init__(self, model, covariance=None, covar_axis=None, fit_range=None,
                 statname=None, statval=None, npred=None, fluxes=None,
                 flux_errors=None):

        self.model = model
        self.covariance = covariance
        self.covar_axis = covar_axis
        self.fit_range = fit_range.to('TeV')
        self.statname = statname
        self.statval = statval
        self.npred = npred
        self.fluxes = fluxes
        self.flux_errors = flux_errors

    @classmethod
    def from_yaml(cls, filename):
        """Create cls from YAML file

        Parameters
        ----------
        filename : str, Path
            File to read
        """
        filename = make_path(filename)
        val = read_yaml(str(filename))
        return cls.from_dict(val)

    def to_yaml(self, filename, mode='w'):
        """Write YAML file

        Parameters
        ----------
        filename : str
            File to write
        mode : str
            Write mode
        """
        import yaml

        d = self.to_dict()
        val = yaml.safe_dump(d, default_flow_style=False)

        with open(str(filename), mode) as outfile:
            outfile.write(val)

    def to_dict(self):
        """Convert to dict"""
        val = dict()
        val['model'] = self.model.to_dict()
        if self.fit_range is not None:
            val['fit_range'] = dict(min = str(self.fit_range[0]),
                                    max = str(self.fit_range[1]))
        if self.statval is not None:
            val['statval'] = self.statval
        if self.statname is not None:
            val['statval'] = self.statname
        if self.covariance is not None:
            val['covariance'] = dict(matrix = self.covariance.tolist(),
                                     axis = self.covar_axis)
        return val

    @classmethod
    def from_dict(cls, val):
        try:
            erange = val['fit_range']
            energy_range = (erange['min'], erange['max']) * u.Unit(erange['unit'])
        except KeyError:
            energy_range = None
        pars = val['parameters']
        parameters = Bunch()
        parameter_errors = Bunch()
        for par in pars:
            parameters[par] = pars[par]['value'] * u.Unit(pars[par]['unit'])
            parameter_errors[par] = pars[par]['error'] * u.Unit(pars[par]['unit'])
        spectral_model = val['spectral_model']

        try:
            fl = val['fluxes']
        except KeyError:
            fluxes=None
            flux_errors=None
        else:
            fluxes = Bunch()
            flux_errors = Bunch()
            for flu in fl:
                fluxes[flu] = fl[flu]['value'] * u.Unit(fl[flu]['unit'])
                flux_errors[flu] = fl[flu]['error'] * u.Unit(fl[flu]['unit'])

        return cls(fit_range=energy_range, parameters=parameters,
                   parameter_errors=parameter_errors,
                   spectral_model=spectral_model, fluxes=fluxes,
                   flux_errors=flux_errors)

    def to_table(self, **kwargs):
        t = Table()
        t['model'] = [self.spectral_model]
        for par in self.parameters.keys():
            t[par] = Column(data=np.atleast_1d(self.parameters[par]), **kwargs)
            t['{}_err'.format(par)] = Column(
                data=np.atleast_1d(self.parameter_errors[par]), **kwargs)

        t['fit_range'] = Column(data=[self.fit_range], unit=self.fit_range.unit,
                                **kwargs)
        if self.fluxes is not None:
            t['flux[1TeV]'] = Column(data=np.atleast_1d(self.fluxes['1TeV']),
                                      **kwargs)
            t['flux_err[1TeV]'] = Column(
                data=np.atleast_1d(self.flux_errors['1TeV']), **kwargs)
        return t

    @property
    def model_with_uncertainties(self):
        """Model with uncertainties

        This uses the uncertainties packages as explained here
        https://pythonhosted.org/uncertainties/user_guide.html#use-of-a-covariance-matrix

        Examples
        --------
        TODO
        """
        raise NotImplementedError()

    def __str__(self):
        """
        Summary info string.
        """
        info = 'Fit result info \n'
        info += '--------------- \n'
        info += 'Model: {} \n'.format(self.model)

        if self.statval is not None:
            info += '\nStatistic: {0:.3f} ({1})'.format(self.statval, self.statname)
        if self.covariance is not None:
            info += '\nCovariance:\n {}'.format(self.covariance)
        if self.fit_range is not None:
            info += '\nFit Range: {}'.format(self.fit_range)
        
        return info

    def info(self):
        """
        Print summary info.
        """
        print(str(self))


class SpectrumResult(object):
    """Class holding all results of a spectral analysis
    
    This class is responsible for all debug plots / numbers

    Parameters
    ----------
    fit: `~gammapy.spectrum.results.SpectrumFitResult`
        Spectrum fit result
    obs: `~gammapy.spectrum.SpectrumObservation`, optional
        Observation used for the fit
    points: `~gammapy.spectrum.DifferentialFluxPoints`, optional
        Flux points
    """

    def __init__(self, fit=None, obs=None, points=None):
        self.fit = fit
        self.obs = obs
        self.points = points

    @property
    def expected_on_vector(self):
        """Counts predicted by a model plus background estimate"""
        energy = self.obs.background_vector.energy
        data = (self.obs.background_vector.data.value + self.fit.npred) * u.ct
        idx = np.isnan(data)
        data[idx] = 0
        return CountsSpectrum(data=data, energy=energy)

    def plot_fit(self):
        """Standard debug plot
        
        Plot ON counts in comparison to background estimate plus source
        counts predicted by a model
        """
        from matplotlib import gridspec
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        gs = gridspec.GridSpec(4, 1)

        ax0 = plt.subplot(gs[:-1,:])
        ax1 = plt.subplot(gs[3,:], sharex=ax0)

        gs.update(hspace=0)
        plt.setp(ax0.get_xticklabels(), visible=False)

        self.obs.background_vector.plot(ax=ax0,
                                        label='Background estimate',
                                        fmt=None,
                                        energy_unit='TeV')

        self.expected_on_vector.plot(ax=ax0,
                                     label='Predicted ON counts')

        self.obs.on_vector.plot(ax=ax0,
                                label='Deteced ON counts',
                                show_poisson_errors=True,
                                fmt=None,
                                energy_unit='TeV')

        ax0.legend(numpoints=1)

        res = (self.expected_on_vector.data - self.obs.on_vector.data).value
        resspec = CountsSpectrum(data=res, energy=self.obs.on_vector.energy)
        resspec.plot(ax=ax1, ecolor='black', fmt=None)
        xx = ax1.get_xlim()
        yy = [0, 0]
        ax1.plot(xx, yy, color='black')

        ymax = 1.4 * max(resspec.data)
        ax1.set_ylim(-ymax, ymax)

        xmin = self.fit.fit_range.to('TeV').value[0] * 0.8
        xmax = self.fit.fit_range.to('TeV').value[1] * 1.2
        ax1.set_xlim(xmin, xmax)
        ax1.set_xlabel('E [{}]'.format('TeV'))
        ax1.set_ylabel('ON (Predicted - Detected)')

        return ax0, ax1

    def plot_spectrum(self, energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
                      energy_power=0, fit_kwargs=None, point_kwargs=None):
        """Plot full spectrum including flux points and residuals

        Parameters
        ----------
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int
            Power of energy to multiply flux axis with
        fit_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.SpectrumFitResult.plot`
        point_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.DifferentialFluxPoints.plot`

        Returns
        -------
        ax0 : `~matplolib.axes`
            Spectrum plot axis
        ax1 : `~matplolib.axes`
            Residuals plot axis
        """
        from matplotlib import gridspec
        import matplotlib.pyplot as plt

        gs = gridspec.GridSpec(4, 1)

        ax0 = plt.subplot(gs[:-1,:])
        ax1 = plt.subplot(gs[3,:], sharex=ax0)

        gs.update(hspace=0)
        plt.setp(ax0.get_xticklabels(), visible=False)

        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')

        if fit_kwargs is None:
            fit_kwargs = dict(label='Best Fit {}'.format(
                    self.fit.spectral_model), color='navy', lw=2)
        if point_kwargs is None:
            point_kwargs = dict(color='navy')


        self.fit.plot(energy_unit=energy_unit, flux_unit=flux_unit,
                      energy_power=energy_power, ax=ax0, **fit_kwargs)
        self.points.plot(energy_unit=energy_unit, flux_unit=flux_unit,
                         energy_power=energy_power, ax=ax0, **point_kwargs)
        self._plot_residuals(energy_unit=energy_unit, ax=ax1, **point_kwargs)

        plt.xlim(self.fit.fit_range[0].to(energy_unit).value * 0.9,
                 self.fit.fit_range[1].to(energy_unit).value * 1.1)

        ax0.legend(numpoints=1)
        return ax0, ax1

    def _plot_residuals(self, ax=None, energy_unit='TeV', **kwargs):
        """Plot residuals

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault('fmt', 'o')

        y, y_err = self._calculate_residuals_points()
        x = self.points['ENERGY'].quantity
        x = x.to(energy_unit).value
        ax.errorbar(x, y, yerr=y_err, **kwargs)

        xx = ax.get_xlim()
        yy = [0, 0]
        ax.plot(xx, yy, color='black')

        ax.set_xlabel('E [{}]'.format(energy_unit))
        ax.set_ylabel('Residuals')

        return ax

    def _calculate_residuals_points(self):
        """Calculate residuals and residual errors

        Based on `~gammapy.spectrum.results.SpectrumFitResult` and
        `~gammapy.spectrum.results.FluxPoints`

        Returns
        -------
        residuals : `~astropy.units.Quantity`
            Residuals
        residuals_err : `~astropy.units.Quantity`
            Residual errors
        """
        x = self.points['ENERGY'].quantity
        y = self.points['DIFF_FLUX'].quantity
        y_err = self.points['DIFF_FLUX_ERR_HI'].quantity

        func_y = self.fit.evaluate(x)
        err_y = self.fit.evaluate_butterfly(x)
        residuals = (y - func_y) / y
        # Todo: add correct formular (butterfly)
        residuals_err = np.sqrt(y_err ** 2 + err_y[0] ** 2) / y

        return residuals.decompose(), residuals_err.decompose()

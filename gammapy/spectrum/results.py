from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from collections import OrderedDict

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
    spectral_model : str
        Spectral model
    parameters : dict
        Fitted parameters
    parameter_errors : dict
        Parameter errors
    covariance : array-like
        Full covarianve matrix
    fit_range : `~gammapy.utils.energy.EnergyBounds`
        Energy range of the spectral fit
    statname : str, optional
        Statistic used for the fit
    statval : float, optional
        Final fit statistic
    n_pred : array-like, optional
        On counts predicted by the fit
    fluxes : dict, optional
        Flux for the fitted model at a given energy
    flux_errors : dict, optional
        Error on the flux for the fitted model at a given energy
    """

    HIGH_LEVEL_KEY = 'fit_result'

    def __init__(self, spectral_model, parameters, parameter_errors,
                 covariance=None, fit_range=None, statname=None, statval=None,
                 n_pred=None, fluxes=None, flux_errors=None):

        self.spectral_model = spectral_model
        self.parameters = Bunch(parameters)
        self.parameter_errors = Bunch(parameter_errors)
        self.fit_range = fit_range
        self.covariance = covariance
        self.statname = statname
        self.statval = statval
        self.n_pred = n_pred
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
        return cls.from_dict(val[cls.HIGH_LEVEL_KEY])

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

        d = dict()
        d[self.HIGH_LEVEL_KEY] = self.to_dict()
        val = yaml.safe_dump(d, default_flow_style=False)

        with open(str(filename), mode) as outfile:
            outfile.write(val)

    @classmethod
    def from_3fgl(cls, source):
        """Retrieve spectral fit result from 3FGL source
        
        TODO : Move to gammapy.catalog
        """
        from astropy.units import Quantity
        d = source.data

        parameters = dict()
        parameter_errors = dict()
        spectral_model = d['SpectrumType'].strip()
        if spectral_model == 'PowerLaw':
            parameters['index'] = Quantity(d['Spectral_Index'], '')
            parameter_errors['index'] = Quantity(d['Unc_Spectral_Index'], '')
            parameters['reference'] = Quantity(d['Pivot_Energy'], 'MeV')
            parameter_errors['reference'] = Quantity(0, 'MeV')
            parameters['norm'] = Quantity(d['Flux_Density'], 'cm-2 s-1 MeV-1')
            parameter_errors['norm'] = Quantity(d['Unc_Flux_Density'],
                                                'cm-2 s-1 MeV-1')
        elif spectral_model == 'LogParabola':
            parameters['alpha'] = Quantity(d['Spectral_Index'], '')
            parameter_errors['alpha'] = Quantity(d['Unc_Spectral_Index'], '')
            parameters['beta'] = Quantity(d['beta'], '')
            parameter_errors['beta'] = Quantity(d['Unc_beta'], '')
            parameters['reference'] = Quantity(d['Pivot_Energy'], 'MeV')
            parameter_errors['reference'] = Quantity(0, 'MeV')
            parameters['norm'] = Quantity(d['Flux_Density'], 'cm-2 s-1 MeV-1')
            parameter_errors['norm'] = Quantity(d['Unc_Flux_Density'],
                                                'cm-2 s-1 MeV-1')

        # Todo : Implement models PLExpCutoff, PLSuperExpCutoff
        else:
            raise NotImplementedError(
                'SpectrumFitResult.from_3fgl not available'
                ' for model {}'.format(spectral_model))



        return cls(spectral_model, parameters, parameter_errors)

    @classmethod
    def from_fitspectrum_json(cls, filename, model=0):
        import json

        with open(filename) as fh:
            data = json.load(fh)

        val = data['fit_functions']['fit_functions'][model]
        e_range = (val['energy_min'], val['energy_max'])
        energy_range = EnergyBounds(e_range, 'TeV')
        spectral_model = val['type']
        parameters = Bunch()
        parameters_errors = Bunch()
        for par in val['parameters']:
            pname = par['name']
            if pname == 'Index':
                unit = u.Unit('')
                name = 'index'
            elif pname == 'Norm':
                unit = val['norm_scale'] * u.Unit('cm-2 s-1 TeV-1')
                name = 'norm'
            elif pname == 'E0':
                unit = u.Unit('TeV')
                name = 'reference'
            elif pname == 'Alpha':
                unit = u.Unit('')
                name = 'alpha'
            elif pname == 'Beta':
                unit = u.Unit('')
                name = 'beta'

            else:
                raise ValueError('Unkown Parameter: {}'.format(pname))
            parameters[name] = par['value'] * unit
            parameters_errors[name] = par['error'] * unit

        fluxes = Bunch()
        fluxes['1TeV'] = val['flux_at_1'] * u.Unit('cm-2 s-1 TeV-1')
        flux_errors = Bunch()
        flux_errors['1TeV'] = val['flux_at_1_err'] * u.Unit('cm-2 s-1 TeV-1')

        return cls(fit_range=energy_range, parameters=parameters,
                   parameter_errors=parameters_errors,
                   spectral_model=spectral_model,
                   fluxes=fluxes, flux_errors=flux_errors)

    @classmethod
    def from_sherpa(cls, covar, efilter, model, fitresult):
        """Create `~gammapy.spectrum.results.SpectrumFitResult` from sherpa objects
        """
        from gammapy.spectrum import SpectrumFit

        el, eh = float(efilter.split(':')[0]), float(efilter.split(':')[1])
        energy_range = EnergyBounds((el, eh), 'keV')
        if 'powlaw1d' in model.name:
            spectral_model = 'PowerLaw'
        elif 'logparabola' in model.name:
            spectral_model = 'LogParabola'
        else:
            raise ValueError("Cannot read sherpa model: {}".format(model.name))
        parameters = Bunch()
        parameter_errors = Bunch()

        # Get thawed parameters from covar object (with errors)
        # Todo: Support assymetric errors
        thawed_pars = list()
        for pname, pval, perr in zip(covar.parnames, covar.parvals,
                                     covar.parmaxes):
            pname = pname.split('.')[-1]
            thawed_pars.append(pname)
            factor = 1
            if pname == 'gamma':
                name = 'index'
                unit = u.Unit('')
            elif pname == 'ampl':
                unit = u.Unit('cm-2 s-1 keV-1')
                name = 'norm'
                factor = SpectrumFit.FLUX_FACTOR
            elif pname == 'c1':
                unit = u.Unit('')
                name = 'alpha'
            elif pname == 'c2':
                unit = u.Unit('')
                name = 'beta'
                factor = 1. / np.log(10)
            else:
                raise ValueError('Unkown Parameter: {}'.format(pname))
            parameters[name] = pval * unit * factor
            parameter_errors[name] = perr * unit * factor

        # Get fixed parameters from model (not stored in covar)
        for par in model.pars:
            if par.name in thawed_pars:
                continue
            if par.name == 'ref':
                name = 'reference'
                unit = u.Unit('keV')
            parameters[name] = par.val * unit
            parameter_errors[name] = 0 * unit

        # Assumes that reference value is set to 1TeV
        if not np.isclose(parameters['reference'].to('TeV').value, 1):
            raise ValueError('Reference not at 1 TeV. Flux@1 TeV will be wrong')
        fluxes = Bunch()
        fluxes['1TeV'] = parameters['norm']
        flux_errors = Bunch()
        flux_errors['1TeV'] = parameter_errors['norm']

        npred = model(1)

        return cls(fit_range=energy_range,
                   parameters=parameters,
                   parameter_errors=parameter_errors,
                   covariance=covar.extra_output,
                   spectral_model=spectral_model,
                   statname=fitresult.statname,
                   statval=fitresult.statval,
                   n_pred=npred,
                   fluxes=fluxes,
                   flux_errors=flux_errors)

    def to_dict(self):
        val = dict()
        if self.fit_range is not None:
            val['fit_range'] = self.fit_range.to_dict()
        val['parameters'] = dict()
        for key in self.parameters:
            par = self.parameters[key]
            err = self.parameter_errors[key]
            val['parameters'][key] = dict(value=par.value,
                                          error=err.value,
                                          unit='{}'.format(par.unit))
        val['spectral_model'] = self.spectral_model
        if self.fluxes is not None:
            val['fluxes'] = dict()
            for key in self.fluxes:
                flux = self.fluxes[key]
                flux_err = self.flux_errors[key]
                val['fluxes'][key] = dict(value=flux.value,
                                          error=flux_err.value,
                                          unit='{}'.format(flux.unit))

        # TODO: Save also covar and statval
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

    def to_sherpa_model(self, name='default'):
        """Return sherpa model

        Parameters
        ----------
        name : str, optional
            Name of the sherpa model instance
        """
        import sherpa.models as m

        if self.spectral_model == 'PowerLaw':
            model = m.PowLaw1D('powlaw1d.' + name)
            model.gamma = self.parameters.index.value
            model.ref = self.parameters.reference.to('keV').value
            model.ampl = self.parameters.norm.to('cm-2 s-1 keV-1').value
        elif self.spectral_model == 'LogParabola':
            model = m.LogParabola('logparabola.' + name)
            model.c1 = self.parameters.alpha.value
            # Sherpa models have log10 in the exponent, we want ln
            model.c2 = self.parameters.beta.value * np.log(10)
            model.ref = self.parameters.reference.to('keV').value
            model.ampl = self.parameters.norm.to('cm-2 s-1 keV-1').value
        else:
            raise NotImplementedError(
                'to_sherpa_model for model {}'.format(self.spectral_model))

        return model

    def evaluate(self, x):
        """Wrapper around the evaluate method on the Astropy model classes.

        Parameters
        ----------
        x : `~gammapy.utils.energy.Energy`
            Evaluation point
        """
        if self.spectral_model == 'PowerLaw':
            flux = models.PowerLaw1D.evaluate(x, self.parameters.norm,
                                              self.parameters.reference,
                                              self.parameters.index)
        elif self.spectral_model == 'LogParabola':
            from astropy.units import Quantity
            # LogParabola evaluation does not work with arrays because
            # there is bug when using '**' with Quantities
            # see https://github.com/astropy/astropy/issues/4764
            flux = Quantity([models.LogParabola1D.evaluate(xx,
                                                           self.parameters.norm,
                                                           self.parameters.reference,
                                                           self.parameters.alpha,
                                                           self.parameters.beta)
                             for xx in x])
        else:
            raise NotImplementedError('Not implemented for model {}.'.format(self.spectral_model))

        return flux.to(self.parameters.norm.unit)

    def evaluate_butterfly(self, x, method='analytical'):
        """Calculate butterfly

        Disclaimer: Only available for PowerLaw assuming no correlation
        TODO: Use uncertainties package

        Parameters
        ----------
        x : `~gammapy.utils.energy.Energy`
            Evaluation point
        method : str {'analytical'}
            Computation method

        Returns
        -------
        butterfly : tuple
            Lower, upper flux errors
        """
        if method == 'analytical':
            return self._eval_butterfly_analytical(x)
        else:
            raise NotImplementedError('Method: {}'.format(method))

    def _eval_butterfly_analytical(self, x):
        """Evaluate butterfly using hard-coded formulas"""
        if self.spectral_model == 'PowerLaw':
            from gammapy.spectrum.powerlaw import power_law_df_over_f
            f = self.evaluate(x)
            x = x.to('TeV').value
            e0 = self.parameters.reference.to('TeV').value
            f0 = self.parameters.norm.to('cm-2 s-1 TeV-1').value
            df0 = self.parameter_errors.norm.to('cm-2 s-1 TeV-1').value
            dg = self.parameter_errors.index.value
            # TODO: Fix this!
            cov = 0
            df_over_f = power_law_df_over_f(x, e0, f0, df0, dg, cov)
            val = df_over_f * f
            # Errors are symmetric
            return (val, val)
        else:
            raise NotImplementedError('Analytical butterfly calculation'
                                      ' not implemented for model {}'.format(
                                      self.spectral_model))

    def plot(self, ax=None, energy_unit='TeV', energy_range=None,
             flux_unit='cm-2 s-1 TeV-1', energy_power=0, n_points=100,
             **kwargs):
        """Plot fit function
        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        energy_range : `~gammapy.utils.energy.EnergyBounds`
            Plot range, if fit range not set
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int
            Power of energy to multiply flux axis with
        n_points : int
            Number of evaluation nodes

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
        """

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        xx = self._get_x(energy_range, n_points)
        yy = self.evaluate(xx)
        x = xx.to(energy_unit).value
        y = yy.to(flux_unit).value
        y = y * np.power(x, energy_power)
        flux_unit = u.Unit(flux_unit) * np.power(u.Unit(energy_unit), energy_power)
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        return ax

    def plot_butterfly(self, ax=None, energy_unit='TeV', energy_range=None,
                       flux_unit='cm-2 s-1 TeV-1', energy_power=0,
                       n_points=1000, **kwargs):
        """Plot butterfly

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        energy_range : `~gammapy.utils.energy.EnergyBounds`
            Plot range, if fit range not set
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int
            Power of energy to multiply flux axis with
        n_points : int
            Number of evaluation nodes

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
        """

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        xx = self._get_x(energy_range, n_points)
        yy = self.evaluate(xx)
        y_lo, y_hi = self.evaluate_butterfly(xx)
        y_up = yy + y_hi
        y_down = yy - y_lo

        x = xx.to(energy_unit).value
        y_up = y_up.to(flux_unit).value
        y_down = y_down.to(flux_unit).value

        y_up = y_up * np.power(x, energy_power)
        y_down = y_down * np.power(x, energy_power)
        flux_unit = u.Unit(flux_unit) * np.power(u.Unit(energy_unit), energy_power)
        ax.fill_between(x, y_down, y_up, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        return ax

    def _get_x(self, energy_range, n_points):
        """Helper function to proved x sampling"""
        if energy_range is None:
            if self.fit_range is not None:
                energy_range = self.fit_range
            else:
                raise ValueError('Fit range not set. You have to specify an energy range')
        else:
            energy_range = EnergyBounds(energy_range)

        x_min = np.log10(energy_range[0].to('TeV').value)
        x_max = np.log10(energy_range[1].to('TeV').value)
        x = np.logspace(x_min, x_max, n_points) * u.Unit('TeV')

        return x

    def __str__(self):
        """
        Summary info string.
        """
        info = 'Fit result info \n'
        info += '--------------- \n'
        info += 'Model: {} \n'.format(self.spectral_model)
        info += 'Parameters: \n'

        for name in sorted(self.parameters):
            val = self.parameters[name]
            err = self.parameter_errors[name]
            _ = dict(name=name, val=val, err=err)
            if name == 'norm':
                _['val'] = _['val'].to('1e-12 cm^-2 TeV^-1 s^-1')
                _['err'] = _['err'].to('1e-12 TeV^-1 cm^-2 s^-1')
                info += '\t {name:10s}: ({val.value:.3f} +/- {err.value:.3f}) x {val.unit}\n'.format(**_)
            else:
                info += '\t {name:10s}: {val.value:.3f} +/- {err.value:.3f} {val.unit}\n'.format(**_)

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
        data = (self.obs.background_vector.data.value + self.fit.n_pred) * u.ct
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
                                        energy_unit='TeV')

        self.expected_on_vector.plot(ax=ax0,
                                     show_poisson_errors=True,
                                     label='Predicted ON counts')

        self.obs.on_vector.plot(ax=ax0,
                                label='Deteced ON counts',
                                energy_unit='TeV')

        ax0.legend(numpoints=1)

        res = self.expected_on_vector.data - self.obs.on_vector.data
        resspec = CountsSpectrum(data=res, energy=self.obs.on_vector.energy)
        resspec.plot(ax=ax1, color='black')

        xx = ax1.get_xlim()
        yy = [0, 0]
        ax1.plot(xx, yy, color='black')

        xmin = self.fit.fit_range.to('TeV').value[0] * 0.8
        xmax = self.fit.fit_range.to('TeV').value[1] * 1.2
        ax1.set_xlim(xmin, xmax)
        ax1.set_xlabel('E [{}]'.format('TeV'))
        ax1.set_ylabel('Residuals')

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

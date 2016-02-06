from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from collections import OrderedDict

import numpy as np
from astropy.extern import six
from astropy.table import Table, Column, QTable, hstack, vstack
from astropy.units import Unit

from ..extern.bunch import Bunch
from ..utils.energy import EnergyBounds
from ..utils.scripts import read_yaml


@six.add_metaclass(abc.ABCMeta)
class Result():
    """Base class for spectrum results

    All serialisation methods should be implemented here, all derived classed
    only have from_dict and to_dict methods. The HIGH_LEVEL_KEY describes the
    highest level key in the serialised dicts (that way one can store all
    spectrum results in one file).
    """
    HIGH_LEVEL_KEY = 'Default'

    @classmethod
    def from_dict(cls, val):
        """Create cls from dict

        Parameters
        ----------
        val : dict
            dict to read
        """
        raise NotImplementedError

    @classmethod
    def from_yaml(cls, filename):
        """Create cls from YAML file

        Parameters
        ----------
        filename : str
            File to read
        """
        val = read_yaml(filename)
        return cls.from_dict(val[HIGH_LEVEL_KEY])

    def to_dict(self):
        """Export to Python dict

        The dict can be used for YAML or JSON serialisation.
        """
        raise NotImplementedError

    def to_yaml(self, filename, key):
        """Write YAML file

        Parameters
        ----------
        filename : str
            File to write
        key : str
            Highest level key to read
        """

        import yaml

        d = dict()
        d[HIGH_LEVEL_KEY] = self.to_dict()
        val = yaml.safe_dump(d, default_flow_style=False)

        with open(str(filename), 'w') as outfile:
            outfile.write(val)

    @classmethod
    def from_fitspectrum_json(cls, filename):
        """Read FitSpectrum output file

        TODO: Remove once FitSpectrum writes standard YAML files

        Parameters
        ----------
        filename : str
            Name of the JSON file to read
        """
        raise NotImplementedError

    def to_table(self, **kwargs):
        """Create overview `~astropy.table.Table`

        kwargs are forwarded to `~astropy.table.Column`
        """
        raise NotImplementedError


class SpectrumFitResult(Result):
    """Class representing the result of a spectral fit

    For a complete documentation see :ref:`gadf:spectrum-fit-result`

    Parameters
    ----------
    spectral_model : str
        Spectral model, for allowed names see :ref:`gadf:source-models`
    parameters : dict
        Fitted parameters, for allowed names see :ref:`gadf:source-models`
    parameter_errors : dict
        Parameter errors, for allowed names see :ref:`gadf:source-models`
    energy_range : `~gammapy.utils.energy.EnergyBounds`
        Energy range of the spectral fit
    fluxes : dict, optional
        Flux for the fitted model at a given energy
    flux_errors : dict, optional
        Error on the flux for the fitted model at a given energy
    """

    HIGH_LEVEL_KEY = 'fit_result'

    def __init__(self, spectral_model, parameters, parameter_errors,
                 energy_range=None, fluxes=None, flux_errors=None):

        self.spectral_model = spectral_model
        self.parameters = Bunch(parameters)
        self.parameter_errors = Bunch(parameter_errors)
        self.energy_range = EnergyBounds(energy_range).to('TeV')
        self.fluxes = fluxes
        self.flux_errors = flux_errors

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
                unit = Unit('')
                name = 'index'
            elif pname == 'Norm':
                unit = val['norm_scale'] * Unit('cm-2 s-1 TeV-1')
                name = 'norm'
            elif pname == 'E0':
                unit = Unit('TeV')
                name = 'reference'
            else:
                raise ValueError('Unkown Parameter: {}'.format(pname))
            parameters[name] = par['value'] * unit
            parameters_errors[name] = par['error'] * unit

        fluxes = Bunch()
        fluxes['1TeV'] = val['flux_at_1'] * Unit('cm-2 s-1')
        flux_errors = Bunch()
        flux_errors['1TeV'] = val['flux_at_1_err'] * Unit('cm-2 s-1')

        return cls(energy_range=energy_range, parameters=parameters,
                   parameter_errors=parameters_errors,
                   spectral_model=spectral_model,
                   fluxes=fluxes, flux_errors=flux_errors)

    @classmethod
    def from_sherpa(cls, covar, filter, model):
        """Create `~gammapy.spectrum.results.SpectrumFitResult` from sherpa objects
        """
        el, eh = float(filter.split(':')[0]), float(filter.split(':')[1])
        energy_range = EnergyBounds((el, eh), 'keV')
        if model.type == 'powlaw1d':
            spectral_model = 'PowerLaw'
        else:
            raise ValueError("Model not understood: {}".format(model.name))
        parameters = Bunch()
        parameter_errors = Bunch()

        # Get thawed parameters from covar object (with errors)
        # Todo: Support assymetric errors
        thawed_pars = list()
        for pname, pval, perr in zip(covar.parnames, covar.parvals,
                                     covar.parmaxes):
            pname = pname.split('.')[-1]
            thawed_pars.append(pname)
            if pname == 'gamma':
                name = 'index'
                unit = Unit('')
            elif pname == 'ampl':
                unit = Unit('cm-2 s-1 keV-1')
                name = 'norm'
            else:
                raise ValueError('Unkown Parameter: {}'.format(pname))
            parameters[name] = pval * unit
            parameter_errors[name] = perr * unit

        # Get fixed parameters from model (not stored in covar)
        for par in model.pars:
            if par.name in thawed_pars:
                continue
            if par.name == 'ref':
                name = 'reference'
                unit = Unit('keV')
            parameters[name] = par.val * unit
            parameter_errors[name] = 0 * unit

        fluxes = Bunch()
        fluxes['1TeV'] = model(1e6) * Unit('cm-2 s-1')
        flux_errors = Bunch()
        flux_errors['1TeV'] = 0 * Unit('cm-2 s-1')

        return cls(energy_range=energy_range, parameters=parameters,
                   parameter_errors=parameter_errors,
                   spectral_model=spectral_model,
                   fluxes=fluxes, flux_errors=flux_errors)

    def to_dict(self):
        val = dict()
        val['energy_range'] = self.energy_range.to_dict()
        val['parameters'] = dict()
        for key in self.parameters:
            par = self.parameters[key]
            err = self.parameter_errors[key]
            val['parameters'][key] = dict(value=par.value,
                                          error=err.value,
                                          unit='{}'.format(par.unit))
        val['spectral_model'] = self.spectral_model

        return val

    @classmethod
    def from_dict(cls, val):
        erange = val['energy_range']
        energy_range = (erange['min'], erange['max']) * Unit(erange['unit'])
        pars = val['parameters']
        parameters = Bunch()
        parameter_errors = Bunch()
        for par in pars:
            parameters[par] = pars[par]['value'] * Unit(pars[par]['unit'])
            parameter_errors[par] = pars[par]['error'] * Unit(pars[par]['unit'])
        spectral_model = val['spectral_model']

        return cls(energy_range=energy_range, parameters=parameters,
                   parameter_errors=parameter_errors,
                   spectral_model=spectral_model)

    def to_table(self, **kwargs):
        t = Table()
        for par in self.parameters.keys():
            t[par] = Column(data=np.atleast_1d(self.parameters[par]), **kwargs)
            t['{}_err'.format(par)] = Column(
                data=np.atleast_1d(self.parameter_errors[par]), **kwargs)

        t['e_min'] = Column(data=np.atleast_1d(self.energy_range[0]), **kwargs)
        t['e_max'] = Column(data=np.atleast_1d(self.energy_range[1]), **kwargs)
        if self.fluxes is not None:
            t['flux [1TeV]'] = Column(data=np.atleast_1d(self.fluxes['1TeV']),
                                      **kwargs)
            t['flux_err [1TeV]'] = Column(
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
        else:
            raise NotImplementedError()

        return model

    def plot(self, ax=None, butterfly=True, energy_unit='TeV',
             flux_unit='cm-2 s-1 TeV-1', e_power=0, **kwargs):
        """Plot fit function

        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        e_power : int
            Power of energy to multiply flux axis with

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
        """

        import matplotlib.pyplot as plt
        from gammapy.spectrum import df_over_f

        ax = plt.gca() if ax is None else ax

        func = self.to_sherpa_model()
        x_min = np.log10(self.energy_range[0].value)
        x_max = np.log10(self.energy_range[1].value)
        x = np.logspace(x_min, x_max, 10000) * self.energy_range.unit
        y = func(x.to('keV').value) * Unit('cm-2 s-1 keV-1')

        # Todo: Find better solution
        if butterfly:
            e = x.to('TeV').value
            e0 = self.parameters.reference.to('TeV').value
            f0 = self.parameters.norm.to('cm-2 s-1 TeV-1').value
            df0 = self.parameter_errors.norm.to('cm-2 s-1 TeV-1').value
            dg = self.parameter_errors.index.value
            cov = 0
            e = df_over_f(e, e0, f0, df0, dg, cov) * y
        else:
            e = np.zeros(shape=x.shape()) * Unit(flux_unit)

        x = x.to(energy_unit).value
        y = y.to(flux_unit).value
        e = e.to(flux_unit).value
        y, e = np.asarray([y, e]) * np.power(x, e_power)
        flux_unit = Unit(flux_unit) * np.power(Unit(energy_unit), e_power)
        ax.errorbar(x, y, yerr=e, **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        return ax


class FluxPoints(Table, Result):
    """Flux points table"""

    HIGH_LEVEL_KEY = 'flux_points'

    def from_fitspectrum_json(cls, filename):
        import json
        with open(filename) as fh:
            data = json.load(fh)

        flux_points = Table(data=data['flux_graph']['bin_values'], masked=True)
        flux_points['energy'].unit = 'TeV'
        flux_points['energy_err_hi'].unit = 'TeV'
        flux_points['energy_err_lo'].unit = 'TeV'
        flux_points['flux'].unit = 'cm-2 s-1 TeV-1'
        flux_points['flux_err_hi'].unit = 'cm-2 s-1 TeV-1'
        flux_points['flux_err_lo'].unit = 'cm-2 s-1 TeV-1'

        return cls(flux_points)

    def to_dict(self):
        val = dict(flux_points = dict())
        for col in self:
            val['flux_points'][col] = [float(_) for _ in self[col]]
            val['flux_points']['energy_unit'] = '{}'.format(self['energy'].unit)
            val['flux_points']['flux_unit'] = '{}'.format(self['flux'].unit)

    @classmethod
    def from_dict(cls, val):
        fp = val['flux_points']
        e_unit = Unit(fp.pop('energy_unit'))
        f_unit = Unit(fp.pop('flux_unit'))
        flux_points = Table(val['flux_points'])
        flux_points['energy'].unit = e_unit
        flux_points['energy_err_hi'].unit = e_unit
        flux_points['energy_err_lo'].unit = e_unit
        flux_points['flux'].unit = f_unit
        flux_points['flux_err_hi'].unit = f_unit
        flux_points['flux_err_lo'].unit = f_unit

    def plot(self, ax=None, energy_unit='TeV',
             flux_unit='cm-2 s-1 TeV-1', e_power=0, **kwargs):
        """Plot spectral points

        kwargs are forwarded to :func:`~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        e_power : int
            Power of energy to multiply flux axis with

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        kwargs.setdefault('fmt', 'o')
        ax = plt.gca() if ax is None else ax
        x = self['energy'].quantity.to(energy_unit).value
        y = self['flux'].quantity.to(flux_unit).value
        yh = self['flux_err_hi'].quantity.to(flux_unit).value
        yl = self['flux_err_lo'].quantity.to(flux_unit).value
        y, yh, yl = np.asarray([y, yh, yl]) * np.power(x, e_power)
        flux_unit = Unit(flux_unit) * np.power(Unit(energy_unit), e_power)
        ax.errorbar(x, y, yerr=(yh, yl), **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        return ax


class SpectrumStats(Result):
    """Class summarizing basic spectral parameters

    'Spectrum' refers to a set of on, off, and effective area vectors
     as well as an energy dispersion matrix.

    Parameters
    ----------
    n_on : int
        number of events inside on region
    n_off : int
        number of events inside on region
    alpha : float
        exposure ratio between on and off regions
    excess : float
        number of excess events in on region
    energy_range : `~gammapy.utils.energy.EnergyBounds`
        Energy range over which the spectrum as extracted
    """

    HIGH_LEVEL_KEY = 'spectrum'

    def __init__(self, n_on=None, n_off=None, energy_range=None, **kwargs):
        self.n_on = n_on
        self.n_off = n_off
        self.energy_range = energy_range
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_fitspectrum_json(cls, filename):
        import json

        with open(filename) as fh:
            data = json.load(fh)

        val = data['spectrum_stats']
        return cls(**val)

    @classmethod
    def from_bg_stats_json(cls, filename):
        """Read BgStats json file

        This file can be produces with hap-show

        Parameters
        ----------
        filename : str
            JSON file produced by hap-show
        """
        import json

        with open(filename) as fh:
            data = json.load(fh)

        val = data['rate_stats']['event_stats']
        # Todo: What is the energy range of BgStats?
        val.update(energy_range=EnergyBounds([0.1, 300], 'TeV'))
        return cls(**val)

    def to_dict(self):
        rtval = dict(spectrum=dict())
        v = rtval['spectrum']
        v['n_on'] = int(self.n_on)
        v['n_off'] = int(self.n_off)
        v['alpha'] = float(self.alpha)
        v['excess'] = float(self.excess)
        v['energy_range'] = self.energy_range.to_dict()
        return rtval

    @classmethod
    def from_dict(cls, val):
        d = val['spectrum']

        return cls(**d)

    @classmethod
    def from_yaml(cls, filename):
        """Create `~gammapy.spectrum.results.SpectrumStats` from YAML file

        Parameters
        ----------
        filename : str
            File to read
        """
        val = read_yaml(filename)
        return cls.from_dict(val['spectrumstats'])

    def to_table(self, **kwargs):
        """Create overview `~astropy.table.Table`"""
        t = Table()
        t['n_on'] = Column(data=np.atleast_1d(self.n_on), **kwargs)
        t['n_off'] = Column(data=np.atleast_1d(self.n_off), **kwargs)
        t['alpha'] = Column(data=np.atleast_1d(self.alpha), **kwargs)
        t['excess'] = Column(data=np.atleast_1d(self.excess), **kwargs)
        return t


class SpectrumResult(object):
    """Class holding all results of a spectral analysis

    Parameters
    ----------
    stats: `~gammapy.spectrum.results.SpectrumStats`, optional
        Spectrum stats
    fit: `~gammapy.spectrum.results.SpectrumFitResult`, optional
        Spectrum fit result
    points: `~gammapy.spectrum.results.FluxPoints`, optional
        Flux points
    """

    def __init__(self, **results):

        for k, v in results.items():
            setattr(self, k, v)

    @classmethod
    def from_fitspectrum_json(cls, filename, model=0):
        stats = SpectrumStats.from_fitspectrum_json(filename)
        fit = SpectrumFitResult.from_fitspectrum_json(filename, model=model)
        points = FluxPoints.from_fitspectrum_json(filename)

        return cls(spectrum_stats=stats, spectrum_fit_result=fit,
                   flux_points=points)

    @classmethod
    def from_yaml(cls, filename):
        """Read YAML file

        This method searches the highest-level key in a YAML file and
        creates `~gammapy.spectrum.results.Result` instances depending
        on the available keys
        """
        data = read_yaml(filename)

        results = OrderedDict(fit=None, points=None, stats=None)

        for i, _ in enumerate(cls.__subclasses__()):
            try:
                val = data[_.HIGH_LEVEL_KEY]
            except KeyError:
                pass
            else:
                temp  = _.from_dict(val)
                results[i] = temp
        return cls(**results)

    @classmethod
    def from_all(cls, filename):
        try:
            val = cls.from_fitspectrum_json(filename)
        except ValueError as e1:
            val = cls.from_yaml(filename)

        return val

    def to_table(self, **kwargs):
        """Return `~astropy.table.Table` containing all results"""
        val = [self.stats, self.fit, self.flux_points]
        l = list()
        for result in val:
            if result is not None:
                l.append(result.to_table(**kwargs))
        return hstack(l)

    def calculate_residuals(self):
        """Calculate residuals and residual errors

        Based on `~gammapy.spectrum.results.SpectrumFitResult` and
        `~gammapy.spectrum.results.FluxPoints`

        Returns
        -------
        residuals : `~astropy.units.Quantity`
            Residuals
        residuals_err : `~astropy.units.Quantity`
            Residual errirs
        """
        func = self.fit.to_sherpa_model()
        x = self.flux_points['energy'].quantity
        y = self.flux_points['flux'].quantity
        y_err = self.flux_points['flux_err_hi'].quantity

        func_y = func(x.to('keV').value) * Unit('s-1 cm-2 keV-1')
        residuals = (y - func_y) / y
        # Todo: add correct formular (butterfly)
        residuals_err = y_err / y

        return residuals.decompose(), residuals_err.decompose()

    def plot_spectrum(self, energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
                      e_power=0, fit_kwargs=None, point_kwargs=None):
        """Plot full spectrum including flux points and residuals

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        e_power : int
            Power of energy to multiply flux axis with
        fit_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.results.SpectrumFitResult.plot_fit_function`
        point_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.results.SpectrumFitResult.plot_points`

        Returns
        -------
        ax : `~matplolib.axes`, optional
            Axis
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
            fit_kwargs = dict(label='Best Fit {}'.format(self.spectral_model),
                              barsabove=True, capsize=0, alpha=0.4,
                              color='cornflowerblue', ecolor='cornflowerblue')
        if point_kwargs is None:
            point_kwargs = dict(color='navy')

        self.fit.plot_fit_function(energy_unit=energy_unit, flux_unit=flux_unit,
                               e_power=e_power, ax=ax0, **fit_kwargs)
        self.points.plot_flux_points(energy_unit=energy_unit, flux_unit=flux_unit,
                              e_power=e_power, ax=ax0, **point_kwargs)
        self.plot_residuals(energy_unit=energy_unit, ax=ax1, **point_kwargs)

        plt.xlim(self.energy_range[0].to(energy_unit).value * 0.9,
                 self.energy_range[1].to(energy_unit).value * 1.1)

        ax0.legend(numpoints=1)
        return gs


    def plot_residuals(self, ax=None, energy_unit='TeV', **kwargs):
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

        y, y_err = self.residuals()
        x = self.flux_points['energy'].quantity
        x = x.to(energy_unit).value
        ax.errorbar(x, y, yerr=y_err, **kwargs)

        xx = ax.get_xlim()
        yy = [0, 0]
        ax.plot(xx, yy, color='black')

        ax.set_xlabel('E [{}]'.format(energy_unit))
        ax.set_ylabel('Residuals')

        return ax


class SpectrumResultDict(dict):
    """Dict of several spectrum results

    * `~gammapy.spectrum.results.SpectrumStats`
    * `~gammapy.spectrum.results.SpectrumFitResult`
    """

    def info(self):
        raise NotImplementedError

    @classmethod
    def from_files(cls, files):
        """Create `~gammapy.spectrum.SpectrumResultDict` from a list of files

        Parameters
        ----------
        files : list, tuple
            Files to load
        """
        val = cls()
        for f in files:
            temp = SpectrumResult.from_all(f)
            val[f] = temp
        return val

    def to_table(self, **kwargs):
        """Create overview `~astropy.table.Table`"""

        val = self.keys()
        analyses = Column(val, name='analysis')
        l = list()
        for key in val:
            l.append(self[key].to_table(**kwargs))
        table = vstack(l, join_type='outer')
        table.add_column(analyses, index=0)
        return table

    def overplot_spectra(self):
        """Overplot spectra"""
        raise NotImplementedError




from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table, Column, QTable
from astropy.units import Unit

from gammapy.extern.bunch import Bunch
from gammapy.utils.energy import EnergyBounds
from gammapy.utils.scripts import read_yaml


class SpectrumFitResult(object):
    """Class representing the result of a spectral fit

    * Fit Function
    * Parameters
    * Energy Range
    * Flux points
    """
    def __init__(self, spectral_model, parameters, parameter_errors,
                 energy_range=None, fluxes=None, flux_errors=None,
                 flux_points=None):
        self.spectral_model = spectral_model
        self.parameters = Bunch(parameters)
        self.parameter_errors = Bunch(parameter_errors)
        self.energy_range = EnergyBounds(energy_range).to('TeV')
        self.fluxes = fluxes
        self.flux_errors = flux_errors
        self.flux_points = Table(flux_points)

    @classmethod
    def from_fitspectrum_json(cls, filename, model=0):
        """Read FitSpectrum output file"""
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
                name = 'Gamma'
            elif pname == 'Norm':
                unit = val['norm_scale'] * Unit('cm-2 s-1 TeV-1')
                name = 'Amplitude'
            elif pname == 'E0':
                unit = Unit('TeV')
                name = 'Reference'
            else:
                raise ValueError('Unkown Parameter: {}'.format(pname))
            parameters[name] = par['value'] * unit
            parameters_errors[name] = par['error'] * unit

        fluxes = Bunch()
        fluxes['1TeV'] = val['flux_at_1'] * Unit('cm-2 s-1')
        flux_errors = Bunch()
        flux_errors['1TeV'] = val['flux_at_1_err'] * Unit('cm-2 s-1')
        flux_points = Table(data=data['flux_graph']['bin_values'], masked=True)
        flux_points['energy'].unit = 'TeV'
        flux_points['energy_err_hi'].unit = 'TeV'
        flux_points['energy_err_lo'].unit = 'TeV'
        flux_points['flux'].unit = 'cm-2 s-1 TeV-1'
        flux_points['flux_err_hi'].unit = 'cm-2 s-1 TeV-1'
        flux_points['flux_err_lo'].unit = 'cm-2 s-1 TeV-1'
        flux_points['flux_ul'].unit = 'cm-2 s-1 TeV-1'

        return cls(energy_range=energy_range, parameters=parameters,
                   parameter_errors=parameters_errors,
                   spectral_model=spectral_model,
                   fluxes=fluxes, flux_errors=flux_errors,
                   flux_points=flux_points)

    @classmethod
    def from_sherpa(cls, covar, filter, model):
        """Create SpectrumFitResult from sherpa objects"""
        el, eh = float(filter.split(':')[0]), float(filter.split(':')[1])
        energy_range = EnergyBounds((el, eh), 'keV')
        if model.name.split('.')[0] == 'powlaw1d':
            spectral_model = 'PowerLaw'
        else:
            raise ValueError("Model not understood: {}".format(model.name))
        parameters = Bunch()
        parameter_errors = Bunch()

        # Todo: Support assymetric errors
        for pname, pval, perr in zip(covar.parnames, covar.parvals,
                                     covar.parmaxes):
            pname = pname.split('.')[-1]
            if pname == 'gamma':
                name = 'Gamma'
                unit = Unit('')
            elif pname == 'ampl':
                unit = Unit('cm-2 s-1 keV-1')
                name = 'Amplitude'
            else:
                raise ValueError('Unkown Parameter: {}'.format(pname))
            parameters[name] = pval * unit
            parameter_errors[name] = perr * unit

        fluxes = Bunch()
        fluxes['1TeV'] = model(1e6) * Unit('cm-2 s-1')
        flux_errors = Bunch()
        flux_errors['1TeV'] = 0 * Unit('cm-2 s-1')

        return cls(energy_range=energy_range, parameters=parameters,
                   parameter_errors=parameter_errors,
                   spectral_model=spectral_model,
                   fluxes=fluxes, flux_errors=flux_errors)

    def to_yaml(self):
        import yaml
        val = dict()
        val['energy_range'] = dict(min=self.energy_range[0].value,
                                   max=self.energy_range[1].value,
                                   unit='{}'.format(self.energy_range.unit))
        val['parameters'] = dict()
        for key in self.parameters:
            par = self.parameters[key]
            err = self.parameter_errors[key]
            val['parameters'][key] = dict(value=par.value,
                                          error=err.value,
                                          unit='{}'.format(par.unit))
        val['spectral_model'] = self.spectral_model
        return yaml.safe_dump(val, default_flow_style=False)

    def write(self, filename):
        """Write YAML file

        Floats are rounded
        """
        val = self.to_yaml()
        with open(str(filename), 'w') as outfile:
            outfile.write(val)

    @classmethod
    def from_yaml(cls, val):
        """Create SpectrumResult from YAML dict"""
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

    @classmethod
    def read(cls, filename):
        """Read YAMl file"""
        val = read_yaml(filename)
        return cls.from_yaml(val)

    def to_table(self):
        """Create overview `~astropy.table.QTable`"""
        t = Table()
        for par in self.parameters.keys():
            t[par] = np.atleast_1d(self.parameters[par])
            t['{}_err'.format(par)] = np.atleast_1d(self.parameter_errors[par])

        t['Threshold'] = self.energy_range[0]
        if self.fluxes is not None:
            t['Flux@1TeV'] = self.fluxes['1TeV']
            t['Flux@1TeV_err'] = self.flux_errors['1TeV']
        return t

    def to_sherpa_model(self, name='default'):
        """Return sherpa model"""
        import sherpa.models as m

        if self.spectral_model == 'PowerLaw':
            model = m.PowLaw1D('powlaw1d.' + name)
            model.gamma = self.parameters.Gamma.value
            model.ref = self.parameters.Reference.to('keV').value
            model.ampl = self.parameters.Amplitude.to('cm-2 s-1 keV-1').value
        else:
            raise NotImplementedError()

        return model

    def plot_flux_points(self, ax=None, energy_unit='TeV',
                         flux_unit='cm-2 s-1 TeV-1', e_power=0, **kwargs):
        """Plot spectral points
        """
        import matplotlib.pyplot as plt

        if 'fmt' not in kwargs:
            kwargs.update(fmt='o')
        ax = plt.gca() if ax is None else ax
        x = self.flux_points['energy'].quantity.to(energy_unit).value
        y = self.flux_points['flux'].quantity.to(flux_unit).value
        yh = self.flux_points['flux_err_hi'].quantity.to(flux_unit).value
        yl = self.flux_points['flux_err_lo'].quantity.to(flux_unit).value
        y, yh, yl = np.asarray([y, yh, yl]) * np.power(x, e_power)
        flux_unit = Unit(flux_unit) * np.power(Unit(energy_unit), e_power)
        ax.errorbar(x, y, yerr=(yh, yl), **kwargs)
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Flux [{}]'.format(flux_unit))
        return ax

    def plot_fit_function(self, ax=None, butterfly=True, energy_unit='TeV',
                          flux_unit='cm-2 s-1 TeV-1', e_power=0, **kwargs):
        """Plot fit function points

        Units are hardcoded
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
            e0 = self.parameters.Reference.to('TeV').value
            f0 = self.parameters.Amplitude.to('cm-2 s-1 TeV-1').value
            df0 = self.parameter_errors.Amplitude.to('cm-2 s-1 TeV-1').value
            dg = self.parameter_errors.Gamma.value
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

    def residuals(self):
        """Calculate residuals + errors"""
        func = self.to_sherpa_model()
        x = self.flux_points['energy'].quantity
        y = self.flux_points['flux'].quantity
        y_err = self.flux_points['flux_err_hi'].quantity

        func_y = func(x.to('keV').value) * Unit('s-1 cm-2 keV-1')
        residuals = (y - func_y) / y
        # Todo: add correct formular (butterfly)
        residuals_err = y_err / y

        return residuals.decompose(), residuals_err.decompose()

    def plot_residuals(self, ax=None, energy_unit='TeV', **kwargs):
        """Plot residuals"""
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        if 'fmt' not in kwargs:
            kwargs.update(fmt='o')

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

    def plot_spectrum(self, energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
                      e_power = 0):
        """Plot full spectrum including flux points and residuals"""
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

        self.plot_fit_function(energy_unit=energy_unit, flux_unit=flux_unit,
                               e_power = e_power,
                               label='Best Fit {}'.format(self.spectral_model),
                               barsabove=True, capsize=0, color='cornflowerblue',
                               ecolor='cornflowerblue', alpha=0.4, ax=ax0)

        self.plot_flux_points(energy_unit=energy_unit, flux_unit=flux_unit,
                              e_power=e_power,
                              ax=ax0, color='navy')
        self.plot_residuals(energy_unit=energy_unit, ax=ax1, color='navy')

        plt.xlim(self.energy_range[0].to(energy_unit).value*0.9,
                 self.energy_range[1].to(energy_unit).value*1.1)

        ax0.legend(numpoints=1)
        return gs


class SpectrumFitResultDict(dict):
    """Dict of several `~gammapy.spectrum.SpectrumFitResults`"""

    def info(self):
        pass

    def to_table(self):
        """Create overview `~astropy.table.QTable`"""

        val = self.keys()
        analyses = Column(val, name='Analysis')
        table = self[val.pop(0)].to_table()
        # Add other analysis with correct units
        for key in val:
            temp = QTable(self[key].to_table())
            new_row = list()
            for col in table.columns:
                new_row.append(temp[col].to(table[col].unit).value)
            table.add_row(new_row)

        table.add_column(analyses, index=0)
        return table

    def overplot(self):
        """Overplot spectra"""
        pass




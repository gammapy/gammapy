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
    """
    def __init__(self, spectral_model, parameters, parameter_errors,
                 energy_range=None, fluxes=None, flux_errors=None,
                 flux_points=None):
        self.spectral_model = spectral_model
        self.parameters = parameters
        self.parameter_errors = parameter_errors
        self.energy_range = EnergyBounds(energy_range).to('TeV')
        self.fluxes = fluxes
        self.flux_errors = flux_errors
        self.flux_points = flux_points

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
        t['Flux@1TeV'] = self.fluxes['1TeV']
        t['Flux@1TeV_err'] = self.flux_errors['1TeV']
        return t


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




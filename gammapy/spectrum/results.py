from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table, Column, QTable
from astropy.units import Unit

from gammapy.extern.bunch import Bunch
from gammapy.utils.energy import EnergyBounds


class SpectrumFitResult(object):
    """Class representing the result of a spectral fit

    * Fit Function
    * Parameters
    * Energy Range
    """
    def __init__(self, spectral_model, parameters, parameter_errors,
                 energy_range=None, fluxes=None, flux_errors=None):
        self.spectral_model = spectral_model
        self.parameters = parameters
        self.parameter_errors = parameter_errors
        self.energy_range = energy_range
        self.fluxes = fluxes
        self.flux_errors = flux_errors

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
            name = par['name']
            if name == 'Index':
                unit = Unit('')
                name = 'Gamma'
            elif name == 'Norm':
                unit = val['norm_scale'] * Unit('cm-2 s-1 TeV-1')
                name = 'Amplitude'
            elif name == 'E0':
                unit = Unit('TeV')
                name = 'Reference'
            else:
                raise ValueError('Unkown Parameter: {}'.format(name))
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




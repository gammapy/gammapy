#!/usr/bin/env python
"""Show HESS FitSpectrum JSON file content.

We use this script to check that the JSON exported info
is identical to the one printed and plotted by FitSpectrum
(a HESS internal tool that has a JSON export option).

The classes here could be (yet another) starting point
for `gammapy.spectrum` ... with the JSON serialisation as
just one option (and converters for other formats / classes).

TODO;
- [ ] Export butterfly, energy resolution matrix, bgstats, run-wise spectra and stats
- [ ] Read butterfly, energy resolution matrix, ....
- [ ] Plot spectrum
- [ ] Compute butterfly from covariance matrix here
- [ ] Interface these classes to Sherpa results ... make it easy to compare, i.e. cross-check `hspec` against `FitSpectrum`


"""
import json
import sys
from collections import OrderedDict
import numpy as np
from astropy.table import Table
from gammapy.extern.bunch import Bunch


class DictWithInfo(OrderedDict):
    def info(self, out=None):
        if out is None:
            out = sys.stdout

        out.write('\n*** {} ***:\n'.format(self.name))
        for k, v in self.items():
            out.write('{:30s} : {:30s}\n'.format(str(k), str(v)))


class SpectrumAsciiTableMixin:
    """Mixin class providing a from_ascii method.
    """
    # TODO: should this be a `classmethod` or `staticmethod`?
    @staticmethod
    def from_ascii(filename,
                   energy_unit='TeV',
                   flux_unit='cm^-2 s^-1 TeV^-1'):
        """Read spectrum info from ascii file.

        Expected format has four columns:

        - Energy
        - Flux estimate
        - Flux band lower limit
        - Flux band upper limit

        """
        names = ['Energy', 'Flux', 'Flux_Low', 'Flux_High']
        table = Table.read(filename, format='ascii.basic', names=names)

        table['Energy'].unit = energy_unit
        table['Flux_Low'].unit = flux_unit
        table['Flux'].unit = flux_unit
        table['Flux_High'].unit = flux_unit

        return table


class FitOptions(DictWithInfo):
    """Fit options."""
    # TODO: should this be a `classmethod` or `staticmethod`?
    @staticmethod
    def from_dict(data):
        out = DictWithInfo(data)
        out.name = 'FitOptions'
        return out


class SpectrumStats(DictWithInfo):
    """Spectrum global stats."""
    # TODO: should this be a `classmethod` or `staticmethod`?
    @staticmethod
    def from_dict(data):
        out = DictWithInfo(data)
        out.name = 'SpectrumStats'
        return out


class Spectrum(Table):
    """Spectrum info (counts, exposure, ...)"""
    @classmethod
    def from_json(cls, data):
        table = cls(data=data['bin_values'], masked=True)

        # TODO: not sure if it's useful to do this here!?
        # Mask out spectral bins with zero exposure
        # mask = table

        # Mask out missing values
        # for colname in table.colnames:
        #     NA_MAGIC_VALUE = -999
        #     table[colname].mask = (table[colname] == -NA_MAGIC_VALUE)

        # TODO: set column format for float columns to something like %.3f

        table.meta['energy_axis'] = data['energy_axis']
        table.meta['bins_with_exposure'] = data['bins_with_exposure']
        table.meta['bins_with_safe_energy'] = data['bins_with_safe_energy']
        table.meta['bins_with_counts'] = data['bins_with_counts']

        return table

    @property
    def nonzero_exposure_part(self):
        """Table of bins with exposure (`~astropy.table.Table`)"""
        bins = self.meta['bins_with_exposure']
        table = self[bins[0]: bins[1]]
        return table

    def info(self, out=None):
        if out is None:
            out = sys.stdout

        out.write('\n*** {} ***:\n'.format('Spectrum'))
        for k, v in self.meta.items():
            out.write('{:30s} : {:30s}\n'.format(str(k), str(v)))

        table = self.nonzero_exposure_part
        super(Spectrum, table).info('stats', out=out)
        colnames = ['bin', 'energy_lo', 'energy_hi', 'n_on', 'n_off',
                    'live_time', 'excess', 'background', 'significance']
        table[colnames].pprint(max_lines=-1)
        # super(Spectrum, self).info('stats', out=out)


class FluxPoints(Table, SpectrumAsciiTableMixin):
    """Flux points (energy, flux, ...)."""
    @classmethod
    def from_json(cls, data):
        table = cls(data=data['bin_values'], masked=True)

        # TODO: set column format for float columns to something like %.3f

        table.meta['rebin_parameter'] = data['rebin_parameter']
        table.meta['energy_algorithm'] = data['energy_algorithm']
        table.meta['flux_algorithm'] = data['flux_algorithm']
        table.meta['rebin_algorithm'] = data['rebin_algorithm']
        table.meta['fit_algorithm'] = data['fit_algorithm']
        table.meta['integrate_over_bins'] = data['integrate_over_bins']
        table.meta['number_of_points'] = data['number_of_points']
        table.meta['energy_fit_range'] = [data['energy_fit_range_min'], data['energy_fit_range_max']]

        return table

    def info(self, out=None):
        if out is None:
            out = sys.stdout

        out.write('\n*** {} ***:\n'.format('SpectralPoints'))
        for k, v in self.meta.items():
            out.write('{:30s} : {:30s}\n'.format(str(k), str(v)))

        # table = self.nonzero_exposure_part
        super(FluxPoints, self).info('stats', out=out)
        # columns = ['bin', 'energy_lo', 'energy_hi', 'n_on', 'n_off',
        #            'live_time', 'excess', 'background', 'significance']
        colnames = self.colnames
        self[colnames].pprint(max_lines=-1)
        # super(Spectrum, self).info('stats', out=out)


class SpectralModel:
    """Spectral model base class

    """
    # TODO: should this be a `classmethod` or `staticmethod`?
    @staticmethod
    def from_json(data):
        """Factory function."""
        # Store model-dependent info
        if data['type'] == 'PowerLaw':
            model = SpectralModelPowerLaw() #.from_json(data)
        elif data['type'] == 'ExpCutoffPL3':
            model = SpectralModelExponentialCutoffPowerLaw() # .from_json(data)
        else:
            raise ValueError('Invalid `type`: {}'.format(data['type']))

        # Store model-independent info
        model.meta = Bunch()
        m = model.meta
        m['type'] = data['type']
        m['energy_range'] = [data['energy_min'], data['energy_max']]
        m['flux_at_1'] = data['flux_at_1']
        m['flux_at_1_err'] = data['flux_at_1_err']
        m['flux_above_1'] = data['flux_above_1']
        m['flux_above_1_err'] = data['flux_above_1_err']
        m['flux_point_stats'] = data['flux_point_stats']
        m['norm_scale'] = data['norm_scale']
        m['covariance_matrix'] = data['covariance_matrix']
        m['correlation_matrix'] = data['correlation_matrix']
        m['parameters'] = data['parameters']
        m['fit'] = data['fit']

        return model

    @property
    def covariance_matrix(self):
        """Covariance matrix (`numpy.ndarray`)"""
        d = self.meta['covariance_matrix']
        self._matrix_dict_to_array(d)

    @property
    def correlation_matrix(self):
        """Correlation matrix (`numpy.ndarray`)"""
        d = self.meta['correlation_matrix']
        self._matrix_dict_to_array(d)

    @staticmethod
    def _matrix_dict_to_array(d):
        """Convert matrix dict entry to array

        Parameters
        ----------
        d : `~array_like`
            matrix dict value

        Returns
        -------
        d_array : `~numpy.ndarray`
            matrix array
        """
        return np.array(d)

    def info(self, out=None):
        if out is None:
            out = sys.stdout

        out.write('\n*** {} ***:\n'.format(self.meta['type']))
        # TODO: pretty-print the same info FitFunction does!
        out.write('{}\n'.format(self.meta))


class SpectralModelPowerLaw(SpectralModel):
    """Power-law spectral model.
    """
    def flux(self, energy):
        pars = self.meta['parameters']
        f0 = pars['f0']
        e0 = pars['e0']
        g = pars['g']
        return f0 * (energy / e0) ** (-g)


class SpectralModelExponentialCutoffPowerLaw(SpectralModel):
    """Exponential cutoff power-law spectral model.
    """
    def flux(self, energy):
        pars = self.meta['parameters']
        f0 = pars['f0']
        e0 = pars['e0']
        g = pars['g']
        e_cut = pars['e_cut']
        return f0 * (energy / e0) ** (-g) * np.exp(-(energy / e_cut))


class SpectralModels(list):
    """List of spectral models.

    Has some convenience methods to compare the different models.
    """
    @classmethod
    def from_json(cls, data):
        models = cls()
        for model_data in data['fit_functions']:
            model = SpectralModel.from_json(model_data)
            models.append(model)

        return models

    def info(self, out=None):
        if out is None:
            out = sys.stdout

        out.write('\n*** {} ***:\n'.format('SpectralModels'))
        for model in self:
            out.write('  * {}\n'.format(model.meta.type))
        out.write('\n')
        for model in self:
            model.info(out)

    def get_model(self, name):
        for model in self:
            if model.meta['type'] == name:
                return model
        raise KeyError('Model not found of type: {}'.format(name))

    def test_statistic(self):
        l0 = self.get_model('PowerLaw').meta['fit']['statistic_value']
        l1 = self.get_model('ExpCutoffPL3').meta['fit']['statistic_value']
        ts = 2 * (l1 - l0)
        return ts


class SpectrumResults:
    """Spectrum JSON results file parser and container class.

    At the moment this is simply what's produced by the HESS
    FitSpectrum JSON exporter.
    We probably want to put some more thought into it and
    adapt the FitSpectrum exporter to any improvements we make here.

    Parameters
    ----------
    filename : str
        JSON filename
    """

    def __init__(self, filename):
        with open(filename) as fh:
            self.data = json.load(fh)

        d = self.data
        self.fit_options = FitOptions.from_dict(d['fit_options'])
        self.spectrum_stats = SpectrumStats.from_dict(d['spectrum_stats'])
        self.spectrum = Spectrum.from_json(d['spectrum'])
        self.spectrum_rebinned = Spectrum.from_json(d['spectrum_rebinned'])
        self.flux_points = FluxPoints.from_json(d['flux_graph'])
        self.spectral_models = SpectralModels.from_json(d['fit_functions'])

    def info(self):
        self.fit_options.info()
        self.spectrum_stats.info()
        self.spectrum.info()
        self.spectrum_rebinned.info()
        self.flux_graph.info()
        self.spectral_models.info()


class SpectrumButterfly(SpectrumAsciiTableMixin):
    """Spectrum butterfly.
    """


def main():
    import sys
    filename = sys.argv[1]
    results = SpectrumResults(filename)
    # results.info()
    print('ts = ', results.spectral_models.test_statistic())

if __name__ == '__main__':
    main()

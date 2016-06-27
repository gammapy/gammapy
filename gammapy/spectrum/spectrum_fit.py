# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
from astropy.extern import six
from gammapy.extern.pathlib import Path
from ..utils.energy import Energy
from ..spectrum import CountsSpectrum
from ..spectrum import SpectrumObservationList, SpectrumObservation
from ..data import ObservationTable
from ..utils.scripts import make_path

__all__ = [
    'SpectrumFit',
]

log = logging.getLogger(__name__)


class SpectrumFit(object):
    """
    Spectral Fit

    Parameters
    ----------
    obs_list : SpectrumObservationList
        Observations to fit
    """

    FLUX_FACTOR = 1e-20
    DEFAULT_STAT = 'wstat'

    def __init__(self, obs_list, stat=DEFAULT_STAT):
        if not isinstance(obs_list, SpectrumObservationList):
            raise ValueError('Wrong input format {}\nUse SpectrumObservation'
                             'List'.format(type(obs_list)))

        self.obs_list = obs_list
        self._model = None
        self._thres_lo = None
        self._thres_hi = None
        self.statistic = stat
        self.n_pred = None

    @classmethod
    def from_pha_list(cls, pha_list):
        """Create `~gammapy.spectrum.SpectrumFit` from a list of PHA files

        Parameters
        ----------
        pha_list : list
            list of PHA files
        """
        obs_list = SpectrumObservationList()
        for temp in pha_list:
            f = str(make_path(temp))
            val = SpectrumObservation.read_ogip(f)
            val.meta.phafile = f
            obs_list.append(val)
        return cls(obs_list)

    @classmethod
    def from_observation_table_file(cls, filename):
        """Create `~gammapy.spectrum.SpectrumFit` using a observation table

        Parameters
        ----------
        filename : str
            Observation table file
        """
        obs_table = ObservationTable.read(filename)
        return cls.from_observation_table(obs_table)

    @classmethod
    def from_observation_table(cls, obs_table):
        """Create `~gammapy.spectrum.SpectrumFit` using a `~gammapy.data.ObservationTable`

        Required columns
        - PHAFILE

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
        """
        if not isinstance(obs_table, ObservationTable):
            raise ValueError('Please provide an observation table')

        pha_list = list(obs_table['PHAFILE'])
        return cls.from_pha_list(pha_list)

    @classmethod
    def from_configfile(cls, configfile):
        """Create `~gammapy.spectrum.SpectrumFit` from configfile

        Parameters
        ----------
        configfile : str
            YAML config file
        """
        import yaml
        with open(configfile) as fh:
            config = yaml.safe_load(fh)

        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
        """Create `~gammapy.spectrum.SpectrumFit` using a config dict

        The spectrum extraction step has to have run before
        """
        config = config['fit']
        table_file = 'observation_table.fits'
        obs_table = ObservationTable.read(table_file)
        fit = SpectrumFit.from_observation_table(obs_table)
        fit.model = config['model']
        return fit

    @property
    def model(self):
        """
        Spectral model to be fit
        """
        if self._model is None:
            raise ValueError('No model specified')
        return self._model

    @model.setter
    def model(self, model, name=None):
        """
        Parameters
        ----------
        model : `~sherpa.models.ArithmeticModel`
            Fit model
        name : str
            Name for Sherpa model instance, optional
        """
        import sherpa.models

        name = 'default' if name is None else name

        if isinstance(model, six.string_types):
            if model == 'PL' or model == 'PowerLaw':
                model = sherpa.models.PowLaw1D('powlaw1d.' + name)
                model.gamma = 2
                model.ref = 1e9
                model.ampl = 1e-20
            elif model == 'LOGPAR' or model == 'LogParabola':
                model = sherpa.models.LogParabola('logparabola.' + name)
                model.c1 = 2
                model.c2 = 0
                model.ref = 1e9
                model.ampl = 1e-20
            else:
                raise ValueError("Undefined model string: {}".format(model))

        if not isinstance(model, sherpa.models.ArithmeticModel):
            raise ValueError("Only sherpa models are supported")

        self._model = model

    @property
    def statistic(self):
        """Statistic to be used in the fit"""
        return self._stat

    @statistic.setter
    def statistic(self, stat):
        """Set Statistic to be used in the fit

        Parameters
        ----------
        stat : `~sherpa.stats.Stat`, str
            Statistic
        """
        import sherpa.stats as s

        if isinstance(stat, six.string_types):
            if stat == 'cash':
                stat = s.Cash()
            elif stat == 'wstat':
                stat = s.WStat()
            else:
                raise ValueError("Undefined stat string: {}".format(stat))

        if not isinstance(stat, s.Stat):
            raise ValueError("Only sherpa statistics are supported")

        self._stat = stat

    @property
    def energy_threshold_low(self):
        """
        Low energy threshold of the spectral fit

        If a list of observations is fit at the same time, this is a list with
        the theshold for each observation.
        """
        return self._thres_lo

    @energy_threshold_low.setter
    def energy_threshold_low(self, energy):
        """
        Low energy threshold setter

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`, str
            Low energy threshold
        """
        energy = Energy(energy)
        self._thres_lo = Energy(energy)

    @property
    def energy_threshold_high(self):
        """
        High energy threshold of the spectral fit
        If a list of observations is fit at the same time, this is a list with
        the threshold for each observation.
        """
        return self._thres_hi

    @energy_threshold_high.setter
    def energy_threshold_high(self, energy):
        """
        High energy threshold setter

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`, str
            High energy threshold
        """
        energy = Energy(energy)
        self._thres_hi = Energy(energy)

    @property
    def pha_list(self):
        """Comma-separate list of PHA files"""
        file_list = [str(o.phafile) for o in self.obs_list]
        ret = ','.join(file_list)
        return ret

    def info(self):
        """Print some basic info"""
        ss = 'Model\n'
        ss += str(self.model)
        ss += '\nEnergy Range\n'
        ss += str(self.energy_threshold_low) + ' - ' + str(
            self.energy_threshold_high)
        return ss

    def run(self, method='sherpa', outdir=None):
        """Run all steps

        Parameters
        ----------
        method : str {sherpa}
            Fit method to use
        outdir : Path, str
            directory to write results files to
        """
        cwd = Path.cwd()
        outdir = cwd if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True)
        os.chdir(str(outdir))

        if method == 'sherpa':
            self._run_sherpa_fit()
        else:
            raise ValueError('Undefined fitting method')

        modelname = self.result.spectral_model
        self.result.to_yaml('fit_result_{}.yaml'.format(modelname))
        self.write_npred()

        os.chdir(str(cwd))

    def _run_sherpa_fit(self):
        """Plain sherpa fit using the session object
        """
        from sherpa.astro import datastack
        log.info("Starting SHERPA")
        log.info(self.info())
        ds = datastack.DataStack()
        ds.load_pha(self.pha_list)

        # Make model amplitude O(1e0)
        model = self.model * self.FLUX_FACTOR
        ds.set_source(model)

        namedataset = []
        for i in range(len(ds.datasets)):
            if self.obs_list[i].lo_threshold > self._thres_lo:
                thres_lo = self.obs_list[i].lo_threshold.to('keV').value
            else:
                thres_lo = self._thres_lo.to('keV').value
            if self.obs_list[i].hi_threshold < self._thres_hi:
                thres_hi = self.obs_list[i].hi_threshold.to('keV').value
            else:
                thres_hi = self._thres_hi.to('keV').value
            datastack.notice_id(i + 1, thres_lo, thres_hi)
            namedataset.append(i + 1)
        datastack.set_stat(self.statistic)
        ds.fit(*namedataset)
        datastack.covar(*namedataset)
        covar = datastack.get_covar_results()
        efilter = datastack.get_filter()

        # First go on calculation flux points following
        # http://cxc.harvard.edu/sherpa/faq/phot_plot.html
        # This should be split out and improved
        xx = datastack.get_fit_plot().dataplot.x
        dd = datastack.get_fit_plot().dataplot.y
        ee = datastack.get_fit_plot().dataplot.yerr
        mm = datastack.get_fit_plot().modelplot.y
        src = datastack.get_source()(xx)
        points = dd / mm * src
        errors = ee / mm * src
        flux_graph = dict(energy=xx, flux=points, flux_err_hi=errors,
                          flux_err_lo=errors)

        from gammapy.spectrum.results import SpectrumFitResult
        self.result = SpectrumFitResult.from_sherpa(covar, efilter, self.model)
        ds.clear_stack()
        ds.clear_models()

    def make_npred(self):
        """Create `~gammapy.spectrum.CountsSpectrum` of predicted on counts
        """
        self.n_pred = dict()
        for obs in self.obs_list:
            temp = CountsSpectrum.get_npred(self.result, obs)
            self.n_pred[obs.obs_id] = temp

    def write_npred(self, outdir=None):
        """Write predicted counts PHA file
        """
        outdir = make_path('n_pred') if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True)

        if self.n_pred is None:
            self.make_npred()
        for key, val in self.n_pred.items():
            filename = "npred_run{}.fits".format(key)
            val.write(str(outdir / filename), clobber=True)

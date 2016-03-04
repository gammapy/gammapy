# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.extern import six
from ..utils.energy import Energy
from ..spectrum import CountsSpectrum
from ..spectrum.spectrum_extraction import SpectrumObservationList, SpectrumObservation
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
    def from_observation_table(cls, obs_table):
        """Create `~gammapy.spectrum.SpectrumFit` using a `~gammapy.data.ObservationTable`

        Required columns
        - OBS_ID
        - PHAFILE
        """

        pha_list = list(obs_table['PHAFILE'])
        obs_list = SpectrumObservationList()
        for f in pha_list:
            val = SpectrumObservation.read_ogip(f)
            val.meta.phafile = f
            obs_list.append(val)
        return cls(obs_list)

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
        fit.set_default_thresholds()
        return fit

    @property
    def model(self):
        """
        Spectral model to be fit
        """
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
        shape = len(self.obs_list)
        if energy.shape is ():
            energy = Energy(np.ones(shape=shape) * energy.value, energy.unit)

        if energy.shape[0] is not shape:
            raise ValueError('Dimension to not match: {} {}'.format(
                self.obs_list, energy))

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
        shape = len(self.obs_list)
        if energy.shape is ():
            energy = Energy(np.ones(shape=shape) * energy.value, energy.unit)

        if energy.shape[0] is not shape:
            raise ValueError('Dimensions to not match: {} {}'.format(
                self.obs_list, energy))

        self._thres_hi = Energy(energy)

    @property
    def pha_list(self):
        """Comma-separate list of PHA files"""
        file_list = [o.meta.phafile for o in self.obs_list]
        ret = ','.join(file_list)
        return ret

    def set_default_thresholds(self):
        """Set energy threshold to the value in the PHA headers"""
        lo_thres = [o.meta.safe_energy_range[0] for o in self.obs_list]
        hi_thres = [o.meta.safe_energy_range[1] for o in self.obs_list]
        self.energy_threshold_low = lo_thres
        self.energy_threshold_high = hi_thres

    def info(self):
        """Print some basic info"""
        ss = 'Model\n'
        ss += str(self.model)
        ss += '\nEnergy Range\n'
        ss += str(self.energy_threshold_low) + ' - ' + str(
            self.energy_threshold_high)
        return ss

    def run(self, method='sherpa'):
        if method == 'hspec':
            self._run_hspec_fit()
        elif method == 'sherpa':
            self._run_sherpa_fit()
        else:
            raise ValueError('Undefined fitting method')

        modelname = self.result.spectral_model
        self.result.to_yaml('fit_result_{}.yaml'.format(modelname))
        self.write_npred()

    def _run_sherpa_fit(self):
        """Plain sherpa fit using the session object
        """
        from sherpa.astro import datastack
        log.info("Starting SHERPA")
        log.info(self.info())
        ds = datastack.DataStack()
        ds.load_pha(self.pha_list)
        import IPython; IPython.embed()
        ds.set_source(self.model)
        thres_lo = self.energy_threshold_low.to('keV').value
        thres_hi = self.energy_threshold_high.to('keV').value

        namedataset = []
        for i in range(len(ds.datasets)):
            datastack.notice_id(i + 1, thres_lo[i], thres_hi[i])
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
            self.n_pred[obs.meta.obs_id] = temp

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

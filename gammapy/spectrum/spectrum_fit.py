# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from astropy.extern import six

from ..utils.energy import Energy
from ..data import ObservationTable
from ..utils.scripts import (
    make_path,
)

__all__ = [
    'SpectrumFit',
]

log = logging.getLogger(__name__)


class SpectrumFit(object):
    """
    Spectral Fit

    Parameters
    ----------
    pha : list of str, `~gammapy.extern.pathlib.Path`
        List of PHA files to fit
    """

    DEFAULT_STAT = 'wstat'

    def __init__(self, pha, bkg=None, arf=None, rmf=None, stat=DEFAULT_STAT):

        self.pha = [make_path(f) for f in pha]
        self._model = None
        self._thres_lo = None
        self._thres_hi = None
        self.statistic = stat

    @classmethod
    def from_config(cls, config):
        """Create `~gammapy.spectrum.SpectrumFit` from config file"""
        outdir = make_path(config['general']['outdir'])
        # TODO: this is not a good solution! an obs table should be used
        return cls.from_dir(outdir/'ogip_data')

    @classmethod
    def from_dir(cls, dir):
        """Create `~gammapy.spectrum.SpectrumFit` using directory

        All PHA files in the directory will be used
        """
        dir = make_path(dir)
        pha_list = dir.glob('pha_run*.pha')
        return cls(pha_list)

    @classmethod
    def from_observation_table(cls, obs_table):
        """Create `~gammapy.spectrum.SpectrumFit` using a `~gammapy.data.ObservationTable`

        Required columns
        - OBS_ID
        - PHAFILE
        """
        pha_list = list(obs_table['PHAFILE'])
        return cls(pha_list)

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
        self._thres_lo = Energy(energy)

    @property
    def energy_threshold_high(self):
        """
       High energy threshold of the spectral fit
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
        self._thres_hi = Energy(energy)

    @property
    def pha_list(self):
        """Comma-separate list of PHA files"""
        ret = ''
        for p in self.pha:
            ret += str(p) + ","

        return ret

    def info(self):
        """Print some basic info"""
        ss = 'Model\n'
        ss += str(self.model)
        ss += '\nEnergy Range\n'
        ss += str(self.energy_threshold_low) + ' - ' + str(
                self.energy_threshold_high)
        return ss

    def run(self, method='hspec'):
        if method == 'hspec':
            self._run_hspec_fit()
        elif method == 'sherpa':
            self._run_sherpa_fit()
        else:
            raise ValueError('Undefined fitting method')

    def _run_hspec_fit(self):
        """Run the gammapy.hspec fit
        """

        log.info("Starting HSPEC")
        import sherpa.astro.ui as sau
        from ..hspec import wstat

        sau.set_conf_opt("max_rstat", 100)

        thres_lo = self.energy_threshold_low.to('keV').value
        thres_hi = self.energy_threshold_high.to('keV').value
        sau.freeze(self.model.ref)

        list_data = []
        for pha in self.pha:
            datid = pha.parts[-1][7:12]
            sau.load_data(datid, str(pha))
            sau.notice_id(datid, thres_lo, thres_hi)
            sau.set_source(datid, self.model)
            list_data.append(datid)

        wstat.wfit(list_data)

    def _run_sherpa_fit(self):
        """Plain sherpa fit not using the session object
        """
        from sherpa.astro import datastack
        log.info("Starting SHERPA")
        log.info(self.info())
        ds = datastack.DataStack()
        ds.load_pha(self.pha_list)
        ds.set_source(self.model)
        thres_lo = self.energy_threshold_low.to('keV').value
        thres_hi = self.energy_threshold_high.to('keV').value
        ds.notice(thres_lo, thres_hi)
        datastack.set_stat(self.statistic)
        ds.fit()
        datastack.covar()
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
        self.result = SpectrumFitResult.from_sherpa(covar, efilter, self.model,
                                                    flux_graph)
        ds.clear_stack()
        ds.clear_models()

    def apply_containment(self, fit):
        """Apply correction factor for PSF containment in ON region"""
        cont = self.get_containment()
        pass

    def get_containment(self):
        """Calculate PSF correction factor for containment in ON region"""
        # TODO: do something useful here
        return 1


def run_spectrum_fit_using_config(config):
    """
    Run a 1D spectral analysis using a config dict

    Parameters
    ----------
    config : dict
        Config dict

    Returns
    -------
    fit : `~gammapy.spectrum.SpectrumFit`
        Fit instance
    """

    config = config['fit']
    table_file = config['observation_table']
    obs_table = ObservationTable.read(table_file)
    fit = SpectrumFit.from_observation_table(obs_table)
    fit.model = config['model']
    fit.energy_threshold_low = Energy(config['threshold_low'])
    fit.energy_threshold_high = Energy(config['threshold_high'])
    fit.run(method=config['method'])
    log.info("\n\n*** Fit Result ***\n\n{}\n\n\n".format(fit.result.to_table()))
    outdir = make_path(config['outdir'])
    fit.result.to_yaml(str(outdir / config['result_file']))
    return fit

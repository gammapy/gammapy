# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
from astropy.extern import six
from ..extern.pathlib import Path
from ..spectrum import SpectrumObservationList, SpectrumObservation
from ..utils.scripts import make_path

__all__ = [
    'SpectrumFit',
]

log = logging.getLogger(__name__)


class SpectrumFit(object):
    """
    Spectral Fit using Sherpa

    Disclaimer: Gammapy classes cannot be translated to Sherpa classes yet.
    Therefore the input data must have been written to disk in order to be read
    in directly by Sherpa.

    Parameters
    ----------
    obs_list : SpectrumObservationList
        Observations to fit

    Examples
    --------

    Example how to run a spectral analysis and have a quick look at the results.

    ::

        from gammapy.spectrum import SpectrumObservation, SpectrumFit
        import matplotlib.pyplot as plt

        filename = '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits'
        obs = SpectrumObservation.read(filename)

        fit = SpectrumFit(obs)
        fit.run()
        fit.result.plot_fit()
        plt.show()

    TODO: put output image in gammapy-extra and show it here.
    """

    FLUX_FACTOR = 1e-20
    DEFAULT_STAT = 'wstat'
    DEFAULT_MODEL = 'PowerLaw'

    def __init__(self, obs_list, stat=DEFAULT_STAT, model=DEFAULT_MODEL):
        if isinstance(obs_list, SpectrumObservation):
            obs_list = SpectrumObservationList([obs_list])
        if not isinstance(obs_list, SpectrumObservationList):
            raise ValueError('Wrong input format {}\nUse SpectrumObservation'
                             'List'.format(type(obs_list)))

        self.obs_list = obs_list
        self.model = model
        self.statistic = stat
        self._fit_range = None
        # FIXME : This is only true for one observation
        # The ON and OFF counts sould be stacked for more than on obs
        from gammapy.spectrum import SpectrumResult
        self._result = SpectrumResult(obs=obs_list[0])

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

    @property
    def result(self):
        """`~gammapy.spectrum.SpectrumResult`"""
        return self._result

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
                model.ampl = 1
            elif model == 'LOGPAR' or model == 'LogParabola':
                model = sherpa.models.LogParabola('logparabola.' + name)
                model.c1 = 2
                model.c2 = 0
                model.ref = 1e9
                model.ampl = 1
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
    def fit_range(self):
        """
        Energy range of the fit
        """
        return self._fit_range

    @fit_range.setter
    def fit_range(self, fit_range):
        """
        Energy range of the fit 
        """
        self._fit_range = fit_range

    @property
    def pha_list(self):
        """Comma-separate list of PHA files"""
        file_list = [str(o.phafile) for o in self.obs_list]
        ret = ','.join(file_list)
        return ret

    def __str__(self):
        """String repr"""
        ss = 'Model\n'
        ss += str(self.model)
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

        modelname = self.result.fit.spectral_model
        self.result.fit.to_yaml('fit_result_{}.yaml'.format(modelname))
        os.chdir(str(cwd))

    def _run_sherpa_fit(self):
        """Plain sherpa fit using the session object
        """
        from sherpa.astro import datastack
        log.info("Starting SHERPA")
        log.info(str(self))

        ds = datastack.DataStack()
        ds.load_pha(self.pha_list)

        # Make model amplitude O(1e0)
        model = self.model * self.FLUX_FACTOR
        ds.set_source(model)

        namedataset = []
        for i in range(len(ds.datasets)):
            datastack.ignore_bad(i + 1)
            datastack.ignore_bad(i + 1, 1)
            namedataset.append(i + 1)
        datastack.set_stat(self.statistic)
        ds.fit(*namedataset)
        datastack.covar(*namedataset)
        covar = datastack.get_covar_results()
        efilter = datastack.get_filter()
        fitresult = datastack.get_fit_results()
        model = datastack.get_model()
        # TODO : Calculate Pivot energy

        from gammapy.spectrum.results import SpectrumFitResult
        self._result.fit = SpectrumFitResult.from_sherpa(covar,
                                                         efilter,
                                                         model,
                                                         fitresult)

        ds.clear_stack()
        ds.clear_models()

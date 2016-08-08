# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import astropy.units as u
from astropy.extern import six
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from . import SpectrumObservationList, SpectrumObservation, models

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
    obs : SpectrumObservationList
        Observations to fit
    model : `~gammapy.spectrum.models.SpectralModel`, `~sherpa.models.ArithmeticModel`
        Model to be fit
    stat : str, `~sherpa.stats.Stat` 
        Fit statistic to be used

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
    """Numerical constant to bring Sherpa optimizers in valid range"""
    DEFAULT_STAT = 'wstat'
    """Default statistic to be used for the fit"""

    def __init__(self, obs, model, stat=DEFAULT_STAT):
        if isinstance(obs, SpectrumObservation):
            obs = SpectrumObservationList([obs])
        if not isinstance(obs, SpectrumObservationList):
            raise ValueError('Wrong input format {}\nUse SpectrumObservation'
                             'List'.format(type(obs)))

        self.obs = obs
        self.model = model
        self.statistic = stat
        self._fit_range = None
        from gammapy.spectrum import SpectrumResult
        # TODO : Introduce SpectrumResultList or Dict
        self._result = list()
        for _ in self.obs:
            self._result.append(SpectrumResult(obs=_))

    @property
    def result(self):
        """`~gammapy.spectrum.SpectrumResult`"""
        return self._result

    @property
    def statistic(self):
        """Sherpa `~sherpa.stats.Stat` to be used for the fit"""
        return self._stat

    @statistic.setter
    def statistic(self, stat):
        import sherpa.stats as s

        if isinstance(stat, six.string_types):
            if stat.lower() == 'cstat':
                stat = s.CStat()
            elif stat.lower() == 'wstat':
                stat = s.WStat()
            else:
                raise ValueError("Undefined stat string: {}".format(stat))

        if not isinstance(stat, s.Stat):
            raise ValueError("Only sherpa statistics are supported")

        self._stat = stat

    @property
    def fit_range(self):
        """
        Tuple of `~astropy.units.Quantity`, energy range of the fit
        """
        return self._fit_range

    @fit_range.setter
    def fit_range(self, fit_range):
        self._fit_range = fit_range

    def __str__(self):
        """String repr"""
        ss = 'Model\n'
        ss += str(self.model)
        return ss

    def fit(self):
        """Fit spectrum"""
        from sherpa.fit import Fit
        from sherpa.models import ArithmeticModel

        # Translate model to sherpa model if necessary
        if isinstance(self.model, models.SpectralModel):
            model = self.model.to_sherpa()
        else:
            model = self.model

        if not isinstance(model, ArithmeticModel):
            raise ValueError('Model not understood: {}'.format(model))

        # Make model amplitude O(1e0)
        val = model.ampl.val * self.FLUX_FACTOR ** (-1)
        model.ampl = val

        # TODO : This only works for one obs
        pha = self.obs[0].to_sherpa()

        fit = Fit(pha, model, self.statistic)
        fit_result = fit.fit()


    def run(self, outdir=None):
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

        self.fit()

        # Assume only one model is fit to all data
        modelname = self.result[0].fit.model.__class__.__name__
        self.result[0].fit.to_yaml('fit_result_{}.yaml'.format(modelname))
        os.chdir(str(cwd))


    # TODO : Put as example on RTD
    def _run_sherpa_fit(self):
        """Plain sherpa fit using the session object
        """
        from sherpa.astro import datastack
        from sherpa.utils.err import IdentifierErr

        log.info(str(self))
        ds = datastack.DataStack()
        ds.load_pha(self.pha_list)

        ds.set_source(self.model)

        # Take into account fit range
        if self.fit_range is not None:
            log.info('Restricting fit range to {}'.format(self.fit_range))
            notice_min = self.fit_range[0].to('keV').value
            notice_max = self.fit_range[1].to('keV').value
            datastack.notice(notice_min, notice_max)

        # Ignore bad is not a stack-enabled function
        for i in range(1, len(ds.datasets) + 1):
            datastack.ignore_bad(i)
            # Ignore bad channels in BKG data (required for WSTAT)
            try:
                datastack.ignore_bad(i, 1)
            except IdentifierErr:
                pass

        datastack.set_stat(self.statistic)
        ds.fit()
        datastack.covar()

        covar = datastack.get_covar_results()
        fitresult = datastack.get_fit_results()
        # Set results for each dataset separately 
        from gammapy.spectrum.results import SpectrumFitResult
        for i in range(1, len(ds.datasets) + 1):
            model = datastack.get_model(i)
            efilter = datastack.get_filter(i)
            # TODO : Calculate Pivot energy
            self.result[i - 1].fit = _sherpa_to_fitresult(model, covar,
                                                          efilter, fitresult)

        ds.clear_stack()
        ds.clear_models()


def _sherpa_to_fitresult(shmodel, covar, efilter, fitresult):
    """Create `~gammapy.spectrum.SpectrumFitResult` from Sherpa objects"""

    from . import SpectrumFitResult

    # Translate sherpa model to GP model
    # This is done here since the FLUXFACTOR needs to be applied
    amplfact = SpectrumFit.FLUX_FACTOR
    pardict = dict(gamma=['index', u.Unit('')],
                   ref=['reference', u.keV],
                   ampl=['amplitude', amplfact * u.Unit('cm-2 s-1 keV-1')])
    kwargs = dict()

    for par in shmodel.pars:
        name = par.name
        kwargs[pardict[name][0]] = par.val * pardict[name][1]

    if 'powlaw1d' in shmodel.name:
        model = models.PowerLaw(**kwargs)
    else:
        raise NotImplementedError(str(shmodel))

    covariance = covar.extra_output
    covar_axis = list()
    for par in covar.parnames:
        name = par.split('.')[-1]
        covar_axis.append(pardict[name][0])

    # Apply flux factor to covariance matrix
    idx = covar_axis.index('amplitude')
    covariance[idx] = covariance[idx] * amplfact
    covariance[:, idx] = covariance[:, idx] * amplfact

    temp = efilter.split(':')
    fit_range = [float(temp[0]), float(temp[1])] * u.keV

    npred = shmodel(1)
    statname = fitresult.statname
    statval = fitresult.statval

    # TODO: Calc Flux@1TeV + Error
    return SpectrumFitResult(model=model,
                             covariance=covariance,
                             covar_axis=covar_axis,
                             fit_range=fit_range,
                             statname=statname,
                             statval=statval,
                             npred=npred
                             )

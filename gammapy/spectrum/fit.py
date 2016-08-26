from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import astropy.units as u
from astropy.extern import six
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from . import (
    SpectrumObservationList,
    SpectrumObservation,
    models,
    DifferentialFluxPoints,
)

__all__ = [
    'SpectrumFit',
]

log = logging.getLogger(__name__)


class SpectrumFit(object):
    """
    Spectral Fit using Sherpa

    This is a wrapper around `~sherpa.fit.Fit` that takes care about
    translating gammapy classes to sherpa classes and handling various aspects
    of the fitting correctly. For usage examples see :ref:`spectral_fitting`

    Parameters
    ----------
    obs_list : SpectrumObservationList
        Observations to fit
    model : `~gammapy.spectrum.models.SpectralModel`, `~sherpa.models.ArithmeticModel`
        Model to be fit
    stat : str, `~sherpa.stats.Stat`
        Fit statistic to be used
    """
    FLUX_FACTOR = 1e-20
    """Numerical constant to make model amplitude O(1) during the fit"""
    DEFAULT_STAT = 'wstat'
    """Default statistic to be used for the fit"""

    def __init__(self, obs_list, model, stat=DEFAULT_STAT):
        if isinstance(obs_list, SpectrumObservation):
            obs_list = [obs_list]

        self.obs_list = SpectrumObservationList(obs_list)
        self.model = model
        self.statistic = stat
        self._fit_range = None
        from gammapy.spectrum import SpectrumResult
        # TODO : Introduce SpectrumResultList or Dict
        self._result = list()
        for _ in self.obs_list:
            self._result.append(SpectrumResult(obs=_))

    @property
    def result(self):
        """List of `~gammapy.spectrum.SpectrumResult` for each observation"""
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
        self._fit_range = u.Quantity(fit_range)

    def __str__(self):
        """String repr"""
        ss = 'Model\n'
        ss += str(self.model)
        return ss

    def fit(self):
        """Fit spectrum"""
        from sherpa.fit import Fit
        from sherpa.models import ArithmeticModel, SimulFitModel
        from sherpa.astro.instrument import Response1D
        from sherpa.data import DataSimulFit

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

        if self.fit_range is not None:
            log.info('Restricting fit range to {}'.format(self.fit_range))
            fitmin = self.fit_range[0].to('keV').value
            fitmax = self.fit_range[1].to('keV').value

        # Loop over observations
        pha = list()
        folded_model = list()
        nobs = len(self.obs_list)
        for ii in range(nobs):
            temp = self.obs_list[ii].to_sherpa()
            if self.fit_range is not None:
                temp.notice(fitmin, fitmax)
                if temp.get_background() is not None:
                    temp.get_background().notice(fitmin, fitmax)
            temp.ignore_bad()
            if temp.get_background() is not None:
                temp.get_background().ignore_bad()
            pha.append(temp)
            # Forward folding
            resp = Response1D(pha[ii])
            folded_model.append(resp(model) * self.FLUX_FACTOR)

        data = DataSimulFit('simul fit data', pha)
        fitmodel = SimulFitModel('simul fit model', folded_model)

        log.debug(fitmodel)
        fit = Fit(data, fitmodel, self.statistic)

        fitresult = fit.fit()
        log.debug(fitresult)
        # The model instance passed to the Fit now holds the best fit values
        covar = fit.est_errors()
        log.debug(covar)

        for ii in range(nobs):
            efilter = pha[ii].get_filter()
            shmodel = fitmodel.parts[ii]
            self.result[ii].fit = _sherpa_to_fitresult(shmodel, covar, efilter, fitresult)

    def compute_fluxpoints(self, binning):
        """Compute `~DifferentialFluxPoints` for best fit model

        Calls :func:`~gammapy.spectrum.DifferentialFluxPoints.compute`.

        Parameters
        ----------
        binning : `~astropy.units.Quantity`
            Energy binning, see
            :func:`~gammapy.spectrum.utils.calculate_flux_point_binning` for a
            method to get flux points with a minimum significance.
        """
        # TODO: Think of a way to not store flux points for each observation
        obs_list = self.obs_list
        for res in self.result:
            model = res.fit.model
            res.points = DifferentialFluxPoints.compute(model=model,
                                                        binning=binning,
                                                        obs_list=obs_list)

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


def _sherpa_to_fitresult(shmodel, covar, efilter, fitresult):
    """Create `~gammapy.spectrum.SpectrumFitResult` from Sherpa objects"""

    from . import SpectrumFitResult

    # Translate sherpa model to GP model
    # This is done here since the FLUXFACTOR needs to be applied
    amplfact = SpectrumFit.FLUX_FACTOR
    # Put cutoff parameter back to keV to have constistent units 
    lambdafact = 1e-9
    pardict = dict(gamma=['index', u.Unit('')],
                   ref=['reference', u.keV],
                   ampl=['amplitude', amplfact * u.Unit('cm-2 s-1 keV-1')],
                   cutoff=['lambda_', lambdafact * u.Unit('keV-1')])
    kwargs = dict()

    for par in shmodel.pars:
        name = par.name
        kwargs[pardict[name][0]] = par.val * pardict[name][1]

    if 'powlaw1d' in shmodel.name:
        model = models.PowerLaw(**kwargs)
    elif 'ecpl' in shmodel.name:
        model = models.ExponentialCutoffPowerLaw(**kwargs)
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
    
    # Adjust covariance term for parameter lambda in ExponentialCutoffPowerLaw
    if 'ecpl' in shmodel.name:
        lambda_idx = covar_axis.index('lambda_')
        covariance[lambda_idx] = covariance[lambda_idx] * lambdafact
        covariance[:, lambda_idx] = covariance[:, lambda_idx] * lambdafact

    # Efilter sometimes contains ','
    if ':' in efilter:
        temp = efilter.split(':')
    else:
        temp = efilter.split(',')
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

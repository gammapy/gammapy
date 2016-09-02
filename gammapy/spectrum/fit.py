from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import copy
import numpy as np
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
        self._result = list()
        self._global_result = list()

    @property
    def result(self):
        """List of `~gammapy.spectrum.SpectrumFitResult` for each observation"""
        return self._result

    @property
    def global_result(self):
        """Global `~gammapy.spectrum.SpectrumFitResult`
        
        Contains only model and fitrange over all observations
        """
        return self._global_result

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

        # Reset results
        self._result = list()

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
            log.debug('Noticed channels obs {}: {}'.format(
                ii, temp.get_noticed_channels()))
            # Forward folding
            resp = Response1D(pha[ii])
            folded_model.append(resp(model) * self.FLUX_FACTOR)

        if (len(pha) == 1 and len(pha[0].get_noticed_channels()) == 1):
            raise ValueError('You are trying to fit one observation in only '
                             'one bin, error estimation will fail')

        data = DataSimulFit('simul fit data', pha)
        log.debug(data)
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
            # Skip observations not participating in the fit
            if efilter != '':
                shmodel = fitmodel.parts[ii]
                result = _sherpa_to_fitresult(shmodel, covar, efilter, fitresult)
                result.obs = self.obs_list[ii]
            else:
                result = None
            self._result.append(result)

        valid_result = np.nonzero(self.result)[0][0]
        global_result = copy.deepcopy(self.result[valid_result])
        global_result.npred = None
        global_result.obs = None
        all_fitranges = [_.fit_range for _ in self._result if _ is not None] 
        fit_range_min = min([_[0] for _ in all_fitranges])
        fit_range_max = max([_[1] for _ in all_fitranges]) 
        global_result.fit_range = u.Quantity((fit_range_min, fit_range_max))
        self._global_result = global_result

    def compute_fluxpoints(self, binning):
        """Compute `~DifferentialFluxPoints` for best fit model

        Calls :func:`~gammapy.spectrum.DifferentialFluxPoints.compute`.

        Parameters
        ----------
        binning : `~astropy.units.Quantity`
            Energy binning, see
            :func:`~gammapy.spectrum.utils.calculate_flux_point_binning` for a
            method to get flux points with a minimum significance.

        Returns
        -------
        result : `~gammapy.spectrum.SpectrumResult`
        """
        from . import SpectrumResult
        obs_list = self.obs_list
        model = self.result[0].model
        points = DifferentialFluxPoints.compute(model=model,
                                                binning=binning,
                                                obs_list=obs_list)
        return SpectrumResult(fit=self.result[0], points=points)

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
        modelname = self.result[0].model.__class__.__name__
        self.result[0].to_yaml('fit_result_{}.yaml'.format(modelname))
        os.chdir(str(cwd))


def _sherpa_to_fitresult(shmodel, covar, efilter, fitresult):
    """Create `~gammapy.spectrum.SpectrumFitResult` from Sherpa objects"""

    from . import SpectrumFitResult

    # Translate sherpa model to GP model
    # Units will be transformed to TeV, s, and m to avoid numerical issues
    # e.g. a flux error of O(-13) results in a covariance entry of O(-45) due
    # to the sqrt and unit keV which kills the uncertainties package
    amplfact = SpectrumFit.FLUX_FACTOR
    pardict = dict(gamma=['index', u.Unit('')],
                   ref=['reference',
                        (1 * u.keV).to('TeV')],
                   ampl=['amplitude',
                         (amplfact * u.Unit('cm-2 s-1 keV-1')).to('m-2 s-1 TeV-1')],
                   cutoff=['lambda_',
                           u.Unit('TeV-1')])
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

    # Adjust parameters in covariance matrix
    covariance = copy.deepcopy(covar.extra_output)
    covar_axis = list()
    for idx, par in enumerate(covar.parnames):
        name = par.split('.')[-1]
        covar_axis.append(pardict[name][0])
        temp = covariance[idx] * pardict[name][1]
        covariance[idx] = temp
        temp2 = covariance[:,idx] * pardict[name][1]
        covariance[:,idx] = temp2

    # Efilter sometimes contains ','
    if ':' in efilter:
        temp = efilter.split(':')
    else:
        temp = efilter.split(',')
    fit_range = ([float(temp[0]), float(temp[1])] * u.keV).to('TeV')

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

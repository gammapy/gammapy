from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import copy
import numpy as np
import astropy.units as u
from astropy.extern import six
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from collections import OrderedDict
from . import (
    SpectrumObservationList,
    SpectrumObservation,
    models,
)
import gammapy.stats as stats

__all__ = [
    'SpectrumFit',
]

log = logging.getLogger(__name__)


class SpectrumFit(object):
    """
    Spectral Fit using Sherpa

    TODO: Outdated - update!

    This is a wrapper around `~sherpa.fit.Fit` that takes care about
    translating gammapy classes to sherpa classes and handling various aspects
    of the fitting correctly. For usage examples see :ref:`spectral_fitting`

    Parameters
    ----------
    obs_list : SpectrumObservationList
        Observations to fit, TODO: rename to data
    model : `~gammapy.spectrum.models.SpectralModel`, `~sherpa.models.ArithmeticModel`
        Model to be fit
    stat : str, `~sherpa.stats.Stat`
        Fit statistic to be used
    method : str, `~sherpa.optmethods.OptMethod`
        Fit statistic to be used
    """
    def __init__(self, obs_list, model, stat='wstat', method='sherpa',
                 forward_folded=True, fit_range=None):
        # For internal coherence accept only SpectrumObservationList
        # TODO: add fancy converters to accept also e.g. CountsSpectrum
        if not isinstance(obs_list, SpectrumObservationList):
            raise ValueError('obs_list is not a SpectrumObservationList')

        self.obs_list = obs_list
        self.model = model
        self.stat = stat
        self.method = method
        self.forward_folded = forward_folded
        self.fit_range=fit_range

        # TODO: Reexpose as properties to import docs
        self.predicted_counts = None
        self.statval = None
        self.result = list()
        self.global_result = list()

    def __str__(self):
        """String repr"""
        ss = self.__class__.__name__
        ss += '\nData {}'.format(self.obs_list)
        ss += '\nSource model {}'.format(self.model)
        ss += '\nStat {}'.format(self.stat)
        ss += '\nMethod {}'.format(self.method)
        ss += '\nForward Folded {}'.format(self.forward_folded)
        ss += '\nFit range {}'.format(self.fit_range)

        return ss

    def predict_counts(self, **kwargs):
        """Predict counts for all observations
        """
        predicted_counts = list()
        for data_ in self.obs_list:
            binning = data_.e_reco
            temp = self._predict_counts_helper(binning,
                                               forward_folded=self.forward_folded,
                                               **kwargs)
            predicted_counts.append(temp)
        self.predicted_counts = predicted_counts

    def _predict_counts_helper(self, binning, forward_folded=True, **kwargs):
        """Predict counts for one observation
        
        TODO: Take model as input to reuse for background model

        Returns
        ------
        predicted_counts: `np.array`
            Predicted counts for one observation
        """
        if forward_folded:
            raise NotImplementedError()
        else:
            counts = self.model.integral(binning[:-1], binning[1:])

        # TODO: Check that counts has correct unit ('' or 'ct')
        return counts

    def calc_statval(self):
        """Calc statistic for all observations"""
        statval = list()
        for data_, npred in zip(self.obs_list, self.predicted_counts):
            temp = self._calc_statval_helper(data_, npred)
            statval.append(temp)
        self.statval = statval

    def _calc_statval_helper(self, data, prediction):
        if self.stat == 'wstat':
            raise NotImplementedError()
        elif self.stat == 'cash':
            statsval = stats.cash(n_on=data.on_vector.data.data.value,
                                  mu_on=prediction)
        else:
            raise NotImplementedError('{}'.format(self.stat))
        return np.sum(statsval)

    def fit(self, **kwargs):
        """Run the fit""" 
        if self.method == 'sherpa':
            self._fit_sherpa(**kwargs)
        else:
            raise NotImplementedError('{}'.format(self.method))

    def _fit_sherpa(self):
        """Wrapper around sherpa minimizer
        
        The sherpa data and model call the corresponding methods on
        `~gammapy.spectrum.SpectrumFit`` 
        """
        from sherpa.fit import Fit
        from sherpa.data import Data1DInt
        from sherpa.stats import Likelihood
        from sherpa.optmethods import NelderMead
        from sherpa.models import ArithmeticModel, Parameter, modelCacher1d


        class SherpaModel(ArithmeticModel):
            """Dummy sherpa model for the `~gammapy.spectrum.SpectrumFit`
            
            Parameters
            ----------
            fit : `~gammapy.spectrum.SpectrumFit`
                Fit instance
            """

            def __init__(self, fit):
                # TODO: add Parameter and ParameterList class
                self.fit = fit
                self.sorted_pars = OrderedDict(**self.fit.model.parameters)
                sherpa_name = 'sherpa_model'
                par_list = list()
                for name, par in self.sorted_pars.items():
                    sherpa_par = Parameter(sherpa_name,
                                           name,
                                           par.value,
                                           units=str(par.unit))
                    setattr(self, name, sherpa_par)
                    par_list.append(sherpa_par)

                ArithmeticModel.__init__(self, sherpa_name, par_list)
                self._use_caching = True
                self.cache = 10
                # TODO: Remove after introduction of proper parameter class
                self.reference.freeze()

            @modelCacher1d
            def calc(self, p, x, xhi=None):
                # Adjust model parameters
                for par, parval in zip(self.sorted_pars, p):
                    par_unit = self.sorted_pars[par].unit
                    self.fit.model.parameters[par] = parval * par_unit
                self.fit.predict_counts(folded=False)
                # Return ones since sherpa does some check on the shape
                return np.ones_like(self.fit.obs_list[0].e_reco)


        class SherpaStat(Likelihood):
            """Dummy sherpa stat for the `~gammapy.spectrum.SpectrumFit`

            Parameters
            ----------
            fit : `~gammapy.spectrum.SpectrumFit`
                Fit instance
            """
            def __init__(self, fit):
                sherpa_name = 'sherpa_stat'
                self.fit = fit
                Likelihood.__init__(self, sherpa_name)

            def _calc(self, data, model, *args, **kwargs):
                self.fit.calc_statval()
                # Sum likelihood over all observations
                total_stat = np.sum(self.fit.statval)
                # sherpa return pattern: total stat, fvec
                return total_stat, None

        binning = self.obs_list[0].e_reco 
        # The data is in principle not used but is still set to the correct
        # value for debugging purposes
        data = self.obs_list[0].on_vector.data.data.value
        data = Data1DInt('Dummy data', binning[:-1].value,
                         binning[1:].value, data)
        # DEBUG
        #from sherpa.models import PowLaw1D
        #from sherpa.stats import Cash
        #model = PowLaw1D('sherpa')
        #model.ref = 0.1
        #fit = Fit(data, model, Cash(), NelderMead())

        # NOTE: We cannot use the Levenbergr-Marquart optimizer in Sherpa
        # because it relies on the fvec return value of the fit statistic (we
        # return None). The computation of fvec is not straightforwad, not just
        # stats per bin. E.g. for a cash fit the sherpa stat computes it
        # according to cstat
        # see https://github.com/sherpa/sherpa/blob/master/sherpa/include/sherpa/stats.hh#L122

        fit = Fit(data, SherpaModel(self), SherpaStat(self), NelderMead())
        print(fit.fit())

    def fit_sherpa(self):
        """Fit spectrum"""
        from sherpa.fit import Fit
        from sherpa.models import ArithmeticModel, SimulFitModel
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

        fit = Fit(data, fitmodel, self.stat, method=self.method_fit)

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
        #TODO: doesn't work .compute() never existed, remove whole method?
        raise NotImplementedError
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
        log.info(self.global_result)

        # Assume only one model is fit to all data
        modelname = self.result[0].model.__class__.__name__
        filename = 'fit_result_{}.yaml'.format(modelname)
        log.info('Writing {}'.format(filename))
        self.result[0].to_yaml(filename)
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
        temp2 = covariance[:, idx] * pardict[name][1]
        covariance[:, idx] = temp2

    # Efilter sometimes contains ','
    if ':' in efilter:
        temp = efilter.split(':')
    else:
        temp = efilter.split(',')

    # Special case only one noticed bin
    if len(temp) == 1:
        fit_range = ([float(temp[0]), float(temp[0])] * u.keV).to('TeV')
    else:
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



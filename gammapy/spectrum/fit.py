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
from .utils import calculate_predicted_counts
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

    @property
    def fit_range(self):
        """Fit range"""
        return self._fit_range

    @fit_range.setter
    def fit_range(self, fit_range):
        self._fit_range = fit_range
        self._apply_fit_range()

    def _apply_fit_range(self):
        """Mark bins within desired fit range for each observation
        
        TODO: Split into smaller functions
        TODO: Could reuse code from PHACountsSpectrum
        TODO: Use True (not 0) to mark good bins
        TODO: Add to EnergyBounds
        """
        self._bins_in_fit_range = list()
        for obs in self.obs_list:
            # Take into account fit range
            energy = obs.e_reco
            valid_range = np.zeros(energy.nbins)

            if self.fit_range is not None:
                idx_lo = np.where(energy < self.fit_range[0])[0]
                valid_range[idx_lo] = 1
                
                idx_hi = np.where(energy[:-1] > self.fit_range[1])[0]
                if len(idx_hi) != 0:
                    idx_hi = np.insert(idx_hi, 0, idx_hi[0] - 1)
                valid_range[idx_hi] = 1

            # Take into account thresholds
            try:
                quality = obs.on_vector.quality
            except AttributeError:
                quality = np.zeros(obs.e_reco.nbins)
            
            # Convolve (see TODO above)
            convolved = np.logical_and(1 - quality, 1 - valid_range)

            self._bins_in_fit_range.append(convolved)

    @property
    def true_fit_range(self):
        """True fit range for each observation"""
        true_range = list()
        for binrange, obs in zip(self._bins_in_fit_range, self.obs_list):
            idx = np.where(binrange)[0]
            e_min = obs.e_reco[idx[0]]
            e_max = obs.e_reco[idx[-1] + 1]
            fit_range = u.Quantity((e_min, e_max))
            true_range.append(fit_range)
        return true_range
            
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
            temp = self._predict_counts_helper(data_)
            predicted_counts.append(temp)
        self.predicted_counts = predicted_counts

    def _predict_counts_helper(self, obs):
        """Predict counts for one observation
        
        TODO: Take model as input to reuse for background model

        Returns
        ------
        predicted_counts: `np.array`
            Predicted counts for one observation
        """
        binning = obs.e_reco
        if self.forward_folded:
            temp = calculate_predicted_counts(model=self.model,
                                              livetime=obs.livetime,
                                              aeff=obs.aeff,
                                              edisp=obs.edisp,
                                              e_reco=binning)
            counts = temp.data.data
        else:
            # TODO: This could also be part of calculate predicted counts
            counts = self.model.integral(binning[:-1], binning[1:])

        # Check count unit (~unit of model amplitude)
        cond = counts.unit.is_equivalent('ct') or counts.unit.is_equivalent('')
        if cond:
            counts = counts.value
        else:
            raise ValueError('Predicted counts {}'.format(counts))

        return counts

    def calc_statval(self):
        """Calc statistic for all observations"""
        statval = list()
        for data_, npred in zip(self.obs_list, self.predicted_counts):
            temp = self._calc_statval_helper(data_, npred)
            statval.append(temp)
        self.statval = statval
        self._restrict_statval()

    def _calc_statval_helper(self, obs, prediction):
        if self.stat == 'cash':
            statsval = stats.cash(n_on=obs.on_vector.data.data.value,
                                  mu_on=prediction)
        elif self.stat == 'wstat':
            kwargs = dict(n_on= obs.on_vector.data.data.value,
                          n_off= obs.off_vector.data.data.value,
                          alpha = obs.alpha,
                          mu_sig=prediction)
            statsval = stats.wstat(**kwargs)
        else:
            raise NotImplementedError('{}'.format(self.stat))

        return statsval

    def _restrict_statval(self):
        """Apply valid range to statval
        """
        restricted_statval = list()
        for statval, valid_range in zip(self.statval, self._bins_in_fit_range):
            val = np.where(valid_range, statval, 0)
            restricted_statval.append(val)
        self.statval = restricted_statval

    def fit(self, **kwargs):
        """Run the fit""" 
        if self.method == 'sherpa':
            self._fit_sherpa(**kwargs)
        else:
            raise NotImplementedError('{}'.format(self.method))
        self._make_fit_result()

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
                self.fit.predict_counts()
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

    def _make_fit_result(self):
        """Bunde fit result into `~gammapy.spectrum.SpectrumFitResult`"""
        from . import SpectrumFitResult
        for idx, obs in enumerate(self.obs_list):
            model = self.model.copy()
            covariance = None
            covar_axis = None

            fit_range = self.true_fit_range[idx]
            statname = self.stat
            statval = np.sum(self.statval[idx])
            npred = self.predicted_counts[idx]
            self.result.append(SpectrumFitResult(
                model=model,
                covariance=covariance,
                covar_axis=covar_axis,
                fit_range=fit_range,
                statname=statname,
                statval=statval,
                npred=npred,
                obs=obs
            ))


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
        raise NotImplementedError

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



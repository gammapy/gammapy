from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import copy
import numpy as np
import astropy.units as u
from astropy.extern import six
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from .utils import CountsPredictor 
from . import (
    SpectrumObservationList,
    SpectrumObservation,
    models,
)
from .. import stats

# This cannot be made a delayed import because the pytest matrix fails if it is
# https://travis-ci.org/gammapy/gammapy/jobs/194204926#L1915
try:
    from .sherpa_utils import SherpaModel, SherpaStat
except ImportError:
    pass

__all__ = [
    'SpectrumFit',
]

log = logging.getLogger(__name__)


class SpectrumFit(object):
    """
    Spectral Fit

    For usage examples see :ref:`spectral_fitting`

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Source model. Should return counts if ``forward_folded`` is False and a flux otherwise
    stat : {'wstat', 'cash'}
        Fit statistic
    forward_folded : bool, default: True
        Fold ``model`` with the IRFs given in ``obs_list``
    fit_range : tuple of `~astropy.units.Quantity`
        Fit range, will be convolved with observation thresholds. If you want to
        control which bins are taken into account in the fit for each
        observations, use :func:`~gammapy.spectrum.SpectrumObservation.qualitiy`
    background_model : `~gammapy.spectrum.model.SpectralModel`, optional
        Background model to be used in cash fits
    method : {'sherpa'}
        Optimization backend for the fit
    err_method : {'sherpa'}
        Optimization backend for error estimation
    """

    def __init__(self, model, stat='wstat', forward_folded=True,
                 fit_range=None, background_model=None,
                 method='sherpa', err_method='sherpa'):
        self.model = model
        self.stat = stat
        self.forward_folded = forward_folded
        self.fit_range = fit_range
        self.background_model = background_model
        self.method = method
        self.err_method = method

        self._bins_in_fit_range = None
        self._predicted_counts = None
        self._statval = None

        self.covar_axis = None
        self.covariance = None
        self.result = None

    @property
    def bins_in_fit_range(self):
        """Bins participating in the fit for each observation"""
        return self._bins_in_fit_range

    @property
    def predicted_counts(self):
        """Current value of predicted counts
        
        For each observation a tuple to counts for the on and off region is
        returned.
        """
        return self._predicted_counts

    @property
    def statval(self):
        """Current value of statval

        For each observation the statval per bin is returned
        """
        return self._statval
        

    def __str__(self):
        """String repr"""
        ss = self.__class__.__name__
        ss += '\nSource model {}'.format(self.model)
        ss += '\nStat {}'.format(self.stat)
        ss += '\nForward Folded {}'.format(self.forward_folded)
        ss += '\nFit range {}'.format(self.fit_range)
        if self.background_model is not None:
            ss += '\nBackground model {}'.format(self.background_model)
        ss += '\nBackend {}'.format(self.method)
        ss += '\nError Backend {}'.format(self.err_method)

        return ss

    def apply_fit_range(self, obs_list):
        """Mark bins within desired fit range for each observation
        """
        self._bins_in_fit_range = list()
        for obs in obs_list:
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

            convolved = np.logical_and(1 - quality, 1 - valid_range)

            self._bins_in_fit_range.append(convolved)

    def true_fit_range(self, obs_list):
        """True fit range for each observation

        True fit range is the fit range set in the
        `~gammapy.spectrum.SpectrumFit` with observation threshold taken into
        account.
        """
        self.apply_fit_range(obs_list)

        true_range = list()
        for binrange, obs in zip(self._bins_in_fit_range, obs_list):
            idx = np.where(binrange)[0]
            e_min = obs.e_reco[idx[0]]
            e_max = obs.e_reco[idx[-1] + 1]
            fit_range = u.Quantity((e_min, e_max))
            true_range.append(fit_range)
        return true_range

    def predict_counts(self, obs_list):
        """Predict counts for all observations

        The result is stored as ``predicted_counts`` attribute
        """
        predicted_counts = list()
        for obs in obs_list:
            mu_sig = self._predict_counts_helper(obs,
                                                 self.model,
                                                 self.forward_folded)
            mu_bkg = None
            if self.background_model is not None:
                # For now, never fold background model with IRFs
                mu_bkg = self._predict_counts_helper(obs,
                                                     self.background_model,
                                                     False)
            counts = [mu_sig, mu_bkg]
            predicted_counts.append(counts)
        self._predicted_counts = predicted_counts

    def _predict_counts_helper(self, obs, model, forward_folded=True):
        """Predict counts for one observation

        Parameters
        ----------
        obs : `~gammapy.spectrum.SpectrumObservation`
            Response functions
        model : `~gammapy.spectrum.SpectralModel`
            Source or background model
        forward_folded : bool, default: True
            Fold model with IRFs

        Returns
        ------
        predicted_counts: `np.array`
            Predicted counts for one observation
        """
        predictor = CountsPredictor(model=model)
        if forward_folded:
            predictor.livetime = obs.livetime
            predictor.aeff = obs.aeff
            predictor.edisp = obs.edisp
        else:
            predictor.e_true = obs.e_reco

        predictor.run()
        counts = predictor.npred.data.data

        # Check count unit (~unit of model amplitude)
        cond = counts.unit.is_equivalent('ct') or counts.unit.is_equivalent('')
        if cond:
            counts = counts.value
        else:
            raise ValueError('Predicted counts {}'.format(counts))

        return counts

    def calc_statval(self, predicted_counts, obs_list):
        """Calc statistic for all observations

        The result is stored as attribute ``statval``, bin outside the fit
        range are set to 0.
        """
        statval = list()
        for obs, npred in zip(obs_list, predicted_counts):
            on_stat, off_stat = self._calc_statval_helper(obs, npred)
            stats = (on_stat, off_stat)
            statval.append(stats)
        self._statval = statval
        self._restrict_statval()

    def _calc_statval_helper(self, obs, prediction):
        """Calculate statval one observation

        Parameters
        ----------
        obs : `~gammapy.spectrum.SpectrumObservation`
            Measured counts
        prediction : tuple of `~np.array`
            Predicted (on counts, off counts)

        Returns
        ------
        statsval : tuple or `~np.array`
            Statval for (on, off)
        """
        # Off stat = 0 by default
        off_stat = np.zeros(obs.e_reco.nbins)
        if self.stat == 'cash':
            if self.background_model is not None:
                mu_on = prediction[0] + prediction[1]
                on_stat = stats.cash(n_on=obs.on_vector.data.data.value,
                                     mu_on=mu_on)
                mu_off = prediction[1] / obs.alpha
                off_stat = stats.cash(n_on=obs.off_vector.data.data.value,
                                      mu_on=mu_off)
            else:
                mu_on = prediction[0]
                on_stat = stats.cash(n_on=obs.on_vector.data.data.value,
                                     mu_on=mu_on)
                off_stat = np.zeros_like(on_stat)

        elif self.stat == 'wstat':
            kwargs = dict(n_on=obs.on_vector.data.data.value,
                          n_off=obs.off_vector.data.data.value,
                          alpha=obs.alpha,
                          mu_sig=prediction[0])
            # Store the result of the profile likelihood as bkg prediction
            mu_bkg = stats.get_wstat_mu_bkg(**kwargs)
            prediction[1] = mu_bkg * obs.alpha
            on_stat = stats.wstat(**kwargs)
            off_stat = np.zeros_like(on_stat)
        else:
            raise NotImplementedError('{}'.format(self.stat))

        return on_stat, off_stat

    @property
    def total_stat(self):
        """Statistic summed over all bins and all observations

        This is what is used for the fit
        """
        total_stat = np.sum(self.statval, dtype=np.float64)
        return total_stat

    def _restrict_statval(self):
        """Apply valid fit range to statval
        """
        restricted_statval = list()
        for statval, valid_range in zip(self.statval, self._bins_in_fit_range):
            # Find bins outside safe range
            idx = np.where(np.invert(valid_range))[0]
            statval[0][idx] = 0
            statval[1][idx] = 0

    def _check_valid_fit(self, obs_list):
        """Helper function to give usefull error messages"""
        # TODO: Check if IRFs are given for forward folding
        if self.stat == 'wstat' and obs_list[0].off_vector is None:
            raise ValueError('Off vector required for WStat fit')

    def likelihood_1d(self, model, parname, parvals):
        """Compute likelihood profile

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel`
            Model to draw likelihood profile for
        parname : str
            Parameter to calculate profile for
        parvals : `~astropy.units.Quantity`
            Parameter values
        """
        likelihood = list()
        self.model = model
        for val in parvals:
            self.model.parameters[parname].value = val
            self.predict_counts()
            self.calc_statval()
            likelihood.append(self.total_stat)
        return np.array(likelihood)

    def plot_likelihood_1d(self, ax=None, **kwargs):
        """Plot 1D likelihood profile

        see :func:`~gammapy.spectrum.SpectrumFit.likelihood_1d`
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        yy = self.likelihood_1d(**kwargs)
        ax.plot(kwargs['parvals'], yy)
        ax.set_xlabel(kwargs['parname'])

        return ax

    def fit(self, obs_list):
        """Run the fit
        
        Parameters
        ----------
        obs_list : `~gammapy.spectrum.SpectrumObservationList`
            Observation to fit
        """
        # TODO: add fancy converters to accept also e.g. CountsSpectrum
        if isinstance(obs_list, SpectrumObservation):
            obs_list = SpectrumObservationList([obs_list])
        if not isinstance(obs_list, SpectrumObservationList):
            raise ValueError('Invalid input {} for parameter obs_list'.format(
                type(obs_list)))

        self._check_valid_fit(obs_list)
        self.apply_fit_range(obs_list)
        if self.method == 'sherpa':
            self._fit_sherpa(obs_list)
        else:
            raise NotImplementedError('{}'.format(self.method))

    def _fit_sherpa(self, obs_list):
        """Wrapper around sherpa minimizer
        """
        from sherpa.fit import Fit
        from sherpa.data import Data1DInt
        from sherpa.optmethods import NelderMead

        binning = obs_list[0].e_reco
        # The sherpa data object is not usued in the fit. It is set to the
        # first observation for debugging purposes, see below
        data = obs_list[0].on_vector.data.data.value
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

        self._sherpa_fit = Fit(data,
                               SherpaModel(self, obs_list),
                               SherpaStat(self, obs_list),
                               NelderMead())
        fitresult = self._sherpa_fit.fit()
        log.debug(fitresult)
        self._make_fit_result(obs_list)

    def _make_fit_result(self, obs_list):
        """Bundle fit results into `~gammapy.spectrum.SpectrumFitResult`

        It is important to copy best fit values, because the error estimation
        will change the model parameters and statval again
        """
        from . import SpectrumFitResult
        results = list()

        model = self.model.copy()
        if self.background_model is not None:

            bkg_model = self.background_model.copy()
        else:
            bkg_model = None

        covariance = None
        covar_axis = None
        statname = self.stat

        for idx, obs in enumerate(obs_list):
            fit_range = self.true_fit_range(obs_list)[idx]
            statval = np.sum(self.statval[idx])
            npred_src = copy.deepcopy(self.predicted_counts[idx][0])
            npred_bkg = copy.deepcopy(self.predicted_counts[idx][1])
            results.append(SpectrumFitResult(
                model=model,
                covariance=covariance,
                covar_axis=covar_axis,
                fit_range=fit_range,
                statname=statname,
                statval=statval,
                npred_src=npred_src,
                npred_bkg=npred_bkg,
                background_model=bkg_model,
                obs=obs
            ))

        self.result = results

    def est_errors(self):
        """Estimate errors"""
        if self.err_method == 'sherpa':
            self._est_errors_sherpa()
        else:
            raise NotImplementedError('{}'.format(self.err_method))
        for res in self.result:
            res.covar_axis = self.covar_axis
            res.covariance = self.covariance
            res.model.parameters.set_parameter_covariance(self.covariance, self.covar_axis)

    def _est_errors_sherpa(self):
        """Wrapper around Sherpa error estimator"""
        covar = self._sherpa_fit.est_errors()
        covar_axis = list()
        for idx, par in enumerate(covar.parnames):
            name = par.split('.')[-1]
            covar_axis.append(name)
        self.covar_axis = covar_axis
        self.covariance = copy.deepcopy(covar.extra_output)

    def compute_fluxpoints(self, binning):
        """Compute `~DifferentialFluxPoints` for best fit model

        TODO: Implement

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
        raise NotImplementedError()

    def run(self, obs_list, outdir=None):
        """Run all steps and write result to disk

        Parameters
        ----------
        obs_list : `~gammapy.spectrum.SpectrumObservationList`, `~gammapy.spectrum.SpectrumObservation`
            Observation(s) to fit
        outdir : Path, str
            directory to write results files to (if given)
        """
        log.info('Running {}'.format(self))

        self.fit(obs_list)
        self.est_errors()

        if outdir is not None:
            outdir = make_path(outdir)
            outdir.mkdir(exist_ok=True, parents=True)

            # Assume only one model is fit to all data
            modelname = self.result[0].model.__class__.__name__
            filename = outdir / 'fit_result_{}.yaml'.format(modelname)
            log.info('Writing {}'.format(filename))
            self.result[0].to_yaml(filename)

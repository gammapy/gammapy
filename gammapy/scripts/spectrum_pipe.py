# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..spectrum import (
    SpectrumEnergyGroupMaker,
    FluxPointEstimator,
    SpectrumExtraction,
    SpectrumFit,
    SpectrumResult,
)
from ..background import ReflectedRegionsBackgroundEstimator

__all__ = [
    'SpectrumAnalysisIACT',
]

log = logging.getLogger(__name__)


class SpectrumAnalysisIACT(object):
    """High-level analysis class to perform a full 1D IACT spectral analysis.

    Observation selection must have happened before.

    For a usage example see :gp-extra-notebook:`spectrum_pipe`

    Config options:

    * outdir : `~gammapy.extern.pathlib.Path`, str
        Output folder, None means no output
    * background : dict
        Forwarded to `~gammapy.background.ReflectedRegionsBackgroundEstimator`
    * extraction : dict
        Forwarded to `~gammapy.spectrum.SpectrumExtraction`
    * fit : dict
        Forwareded to `~gammapy.spectrum.SpectrumFit`
    * fp_binning : `~astropy.units.Quantity`
        Flux points binning

    Parameters
    ----------
    observations : `~gammapy.data.ObservationList`
        Observations to analyse
    config : dict
        Config dict
    """

    def __init__(self, observations, config):
        self.observations = observations
        self.config = config

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n{}'.format(self.observations)
        ss += '\n{}'.format(self.config)
        return ss

    def run(self):
        """Run all steps."""
        log.info(" Running {}".format(self.__class__.__name__))
        self.run_extraction()
        self.run_fit()

    def run_extraction(self):
        """Run all steps for the spectrum extraction."""
        self.background_estimator = ReflectedRegionsBackgroundEstimator(
            obs_list=self.observations,
            **self.config['background'])
        self.background_estimator.run()

        self.extraction = SpectrumExtraction(
            obs_list=self.observations,
            bkg_estimate=self.background_estimator.result,
            **self.config['extraction'])
        if self.config['area_max'] != None:
            self.extraction.define_energy_threshold('area_max', percent=self.config['area_max'])
            log.info(" Using the safe Threshold E>E({}*area_max)".format(self.config['area_max']/100.))

        self.extraction.run(outdir=self.config['outdir'])
        # self.extraction.peek() #BKH - 2017/06/26 : for me it crashs
        print("\n\n")
        self.extraction.print()
        print("\n")

    def run_fit(self):
        """Run all step for the spectrum fit."""
        self.fit = SpectrumFit(
            obs_list=self.extraction.observations,
            **self.config['fit'])
        self.fit.run(outdir=self.config['outdir'])

        # TODO: Don't stack again if SpectrumFit has already done the stacking
        stacked_obs = self.extraction.observations.stack()
        self.egm = SpectrumEnergyGroupMaker(stacked_obs)
        self.egm.compute_groups_fixed(self.config['fp_binning'])

        from scipy.stats import chi2, norm
        sigma = 3
        cl = 1 - 2 * norm.sf(sigma)
        delta_ts = chi2.isf(1 - cl, df=1)
        self.flux_point_estimator = FluxPointEstimator(
            groups=self.egm.groups,
            model=self.fit.result[0].model,
            obs=self.extraction.observations,
            sqrt_ts_threshold=delta_ts)
        self.flux_point_estimator.compute_points()

    @property
    def spectrum_result(self):
        """`~gammapy.spectrum.SpectrumResult`"""
        return SpectrumResult(
            points=self.flux_point_estimator.flux_points,
            model=self.fit.result[0].model
        )

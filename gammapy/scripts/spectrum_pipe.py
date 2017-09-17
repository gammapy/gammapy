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
    nsigma_ul_threshold : float
        Significance for an energy bin for which an UL should be computed
    nsigma_ul : float
        Number of sigma used for the UL computation
    """

    def __init__(self, observations, config, nsigma_ul_threshold=1.5, nsigma_ul=3.):
        self.observations = observations
        self.config = config
        self.nsigma_ul_threshold = nsigma_ul_threshold
        self.nsigma_ul = nsigma_ul

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n{}'.format(self.observations)
        ss += '\n{}'.format(self.config)
        return ss

    def run(self):
        """Run all steps."""
        log.info("Running {}".format(self.__class__.__name__))
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

        self.extraction.run(outdir=self.config['outdir'])

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

        self.flux_point_estimator = FluxPointEstimator(
            groups=self.egm.groups,
            model=self.fit.result[0].model,
            obs=self.extraction.observations,
            nsigma_ul_threshold=self.nsigma_ul_threshold,
            nsigma_ul=self.nsigma_ul
            )
        self.flux_point_estimator.compute_points()

    @property
    def spectrum_result(self):
        """`~gammapy.spectrum.SpectrumResult`"""
        return SpectrumResult(
            points=self.flux_point_estimator.flux_points,
            model=self.fit.result[0].model
        )

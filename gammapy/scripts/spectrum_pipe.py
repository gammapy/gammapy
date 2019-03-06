# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from ..utils.scripts import make_path
from ..spectrum import (
    SpectrumEnergyGroupMaker,
    FluxPointEstimator,
    SpectrumExtraction,
    SpectrumFit,
    SpectrumResult,
)
from ..background import ReflectedRegionsBackgroundEstimator

__all__ = ["SpectrumAnalysisIACT"]

log = logging.getLogger(__name__)


class SpectrumAnalysisIACT:
    """High-level analysis class to perform a full 1D IACT spectral analysis.

    Observation selection must have happened before.

    For a usage example see :gp-notebook:`spectrum_pipe`

    Config options:

    * outdir : `pathlib.Path`, str
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
    observations : `~gammapy.data.Observations`
        Observations to analyse
    config : dict
        Config dict
    """

    def __init__(self, observations, config):
        self.observations = observations
        self.config = config

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.observations)
        ss += "\n{}".format(self.config)
        return ss

    def run(self, optimize_opts=None):
        """Run all steps."""
        log.info("Running {}".format(self.__class__.__name__))
        self.run_extraction()
        self.run_fit(optimize_opts)

    def run_extraction(self):
        """Run all steps for the spectrum extraction."""
        self.background_estimator = ReflectedRegionsBackgroundEstimator(
            observations=self.observations, **self.config["background"]
        )
        self.background_estimator.run()

        self.extraction = SpectrumExtraction(
            observations=self.observations,
            bkg_estimate=self.background_estimator.result,
            **self.config["extraction"]
        )

        self.extraction.run()

    def run_fit(self, optimize_opts=None):
        """Run all step for the spectrum fit."""
        self.fit = SpectrumFit(
            obs_list=self.extraction.spectrum_observations, **self.config["fit"]
        )
        self.fit.run(optimize_opts=optimize_opts)
        modelname = self.fit.result[0].model.__class__.__name__
        filename = make_path(self.config["outdir"]) / "fit_result_{}.yaml".format(
            modelname
        )
        self.fit.result[0].to_yaml(filename=filename)

        # TODO: Don't stack again if SpectrumFit has already done the stacking
        stacked_obs = self.extraction.spectrum_observations.stack()
        self.egm = SpectrumEnergyGroupMaker(stacked_obs)
        self.egm.compute_groups_fixed(self.config["fp_binning"])

        self.flux_point_estimator = FluxPointEstimator(
            groups=self.egm.groups,
            model=self.fit.result[0].model,
            obs=self.extraction.spectrum_observations,
        )
        self.flux_points = self.flux_point_estimator.run()

    @property
    def spectrum_result(self):
        """`~gammapy.spectrum.SpectrumResult`"""
        return SpectrumResult(points=self.flux_points, model=self.fit.result[0].model)

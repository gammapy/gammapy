# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import yaml
from ..utils.scripts import make_path
from ..utils.fitting import Fit
from ..spectrum import (
    FluxPointsEstimator,
    FluxPointsDataset,
    SpectrumExtraction,
    SpectrumDatasetOnOffStacker,
)
from ..background import ReflectedRegionsBackgroundEstimator

__all__ = ["SpectrumAnalysisIACT"]

log = logging.getLogger(__name__)


class SpectrumAnalysisIACT:
    """High-level analysis class to perform a full 1D IACT spectral analysis.

    Observation selection must have happened before.

    Config options:

    * outdir : `pathlib.Path`, str
        Output folder, None means no output
    * background : dict
        Forwarded to `~gammapy.background.ReflectedRegionsBackgroundEstimator`
    * extraction : dict
        Forwarded to `~gammapy.spectrum.SpectrumExtraction`
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

    @property
    def _result_dict(self):
        """Convert to dict."""
        val = dict()
        model = self.config["fit"]["model"]
        val["model"] = model.to_dict()

        fit_range = self.config["fit"].get("fit_range")

        if fit_range is not None:
            val["fit_range"] = dict(
                min=fit_range[0].value,
                max=fit_range[1].value,
                unit=fit_range.unit.to_string("fits"),
            )

        val["statval"] = float(self.fit_result.total_stat)
        val["statname"] = "wstat"

        return val

    def write(self, filename, mode="w"):
        """Write to YAML file.

        Parameters
        ----------
        filename : str
            File to write
        mode : str
            Write mode
        """
        d = self._result_dict
        val = yaml.safe_dump(d, default_flow_style=False)

        with open(str(filename), mode) as outfile:
            outfile.write(val)

    def run_fit(self, optimize_opts=None):
        """Run all step for the spectrum fit."""
        fit_range = self.config["fit"].get("fit_range")
        model = self.config["fit"]["model"]

        for obs in self.extraction.spectrum_observations:
            if fit_range is not None:
                obs.mask_fit = obs.counts.energy_mask(fit_range[0], fit_range[1])
            obs.model = model

        self.fit = Fit(self.extraction.spectrum_observations)
        self.fit_result = self.fit.run(optimize_opts=optimize_opts)

        model = self.config["fit"]["model"]
        modelname = model.__class__.__name__

        model.parameters.covariance = self.fit_result.parameters.covariance

        filename = make_path(self.config["outdir"]) / "fit_result_{}.yaml".format(
            modelname
        )

        self.write(filename=filename)

        obs_stacker = SpectrumDatasetOnOffStacker(self.extraction.spectrum_observations)
        obs_stacker.run()

        datasets_fp = obs_stacker.stacked_obs
        datasets_fp.model = model
        self.flux_point_estimator = FluxPointsEstimator(
            e_edges=self.config["fp_binning"], datasets=datasets_fp
        )
        fp = self.flux_point_estimator.run()
        fp.table["is_ul"] = fp.table["ts"] < 4
        self.flux_points = fp

    @property
    def spectrum_result(self):
        """`~gammapy.spectrum.FluxPointsDataset`"""
        return FluxPointsDataset(
            data=self.flux_points, model=self.fit.datasets.datasets[0].model
        )

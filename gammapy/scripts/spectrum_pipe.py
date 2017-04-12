# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..spectrum import (
    SpectrumObservationList,
    SpectrumEnergyGroupMaker,
    FluxPointEstimator,
    SpectrumExtraction,
    SpectrumFit,
    SpectrumResult,
)
from ..background import ReflectedRegionsBackgroundEstimator
from ..utils.scripts import make_path

__all__ = [
    'SpectrumAnalysisIACT',
]

log = logging.getLogger(__name__)


class SpectrumAnalysisIACT(object):
    """High-level analysis class to perform a full 1D IACT spectral analysis

    Observation selected must have happend before.

    TODO : Extend this to allow for config file driven analysis (and eventually
    a command line tool)

    Parameters
    ----------
    observations : `~gammapy.data.ObservationList`
        Observations to analyse
    background_estimator : `~gammapy.background.ReflectedRegionsBackgroundEstimator`
        Background estimator
    extraction : `~gammapy.spectrum.SpectrumExtraction`
        Spectrum extraction
    fit : `~gammapy.spectrum.SpectrumFit`
        Spectrum fit
    fp_binning : `~astropy.units.Quantity`
        Flux points binning
        TODO: Replace with `~gammapy.spectrum.SpectrumEnergyGroupMaker`
    flux_point_estimator : `~gammapy.spectrum.FluxPointEstimator`
        Flux points estimator
        TODO: Input not used at the moment, fpe is created on the fly
    clobber : False
        Overwrite OGIP files
    stacked : Bool, optional
        Stack observations prior to fitting
    outdir : `~gammapy.extern.pathlib.Path`, optional
        Analysis directory to write files (if given)
    """

    def __init__(self, observations, background_estimator, extraction,
                 fit, fp_binning, flux_point_estimator=None,
                 clobber=False, stacked=False, outdir=None):
        self.outdir = outdir
        self.observations = observations
        self.background_estimator = background_estimator
        self.extraction = extraction
        self.fit = fit
        self.fp_binning = fp_binning
        self.flux_point_estimator = flux_point_estimator

        self.clobber = clobber
        self.stacked = stacked

    @classmethod
    def configure(cls, observations, on_region, model, fp_binning,
                  background_kwargs=dict(),
                  extraction_kwargs=dict(),
                  fit_kwargs=dict(),
                  pipeline_kwargs=dict()):
        """Configure the analysis

        This method takes care of instantiating the analysis classes that
        constitue the spectrum pipeline
        """
        bkg_estimator = ReflectedRegionsBackgroundEstimator(on_region=on_region,
                                                            **background_kwargs)
        extraction = SpectrumExtraction(on_region=on_region,
                                        **extraction_kwargs)
        fit = SpectrumFit(model=model, **fit_kwargs)

        return cls(observations=observations,
                   background_estimator=bkg_estimator, extraction=extraction,
                   fit=fit, fp_binning=fp_binning, **pipeline_kwargs)

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n{}'.format(self.observations)
        ss += '\n{}'.format(self.background_estimator)
        ss += '\n{}'.format(self.fit)
        ss += '\nFlux points binning{}'.format(self.fp_binning)
        ss += '\n{}'.format(self.flux_point_estimator)
        ss += '\nStacked {}'.format(self.stacked)
        return ss

    @property
    def spectrum_observations(self):
        """`~gammapy.spectrum.SpectrumObservationList`"""

        return self._spectrum_observations

    def run(self):
        """Run all steps"""
        if (make_path(self.outdir) / 'ogip_data').is_dir():
            log.info('Reading in OGIP data')
            spec_obs = SpectrumObservationList.read(
                self.outdir / 'ogip_data')
        else:
            self.extract_all()
            spec_obs = self.extraction.observations

        self.fit_all(spec_obs)

    def extract_all(self):
        """Run all steps for the spectrum extraction
        """
        self.background_estimator.run(self.observations)
        self.extraction.run(self.observations,
                            self.background_estimator.result,
                            outdir=self.outdir)

    def fit_all(self, observations):
        """Run all step for the spectrum fit"""
        stacked_obs = observations.stack()
        if self.stacked:
            self.fit.run(stacked_obs, outdir=self.outdir)
        else:
            self.fit.run(observations, outdir=self.outdir)
        egm = SpectrumEnergyGroupMaker(stacked_obs)
        egm.compute_groups_fixed(self.fp_binning)
        self.flux_point_estimator = FluxPointEstimator(
            groups=egm.groups,
            model=self.fit.result[0].model,
            obs=observations)
        self.flux_point_estimator.compute_points()

    @property
    def spectrum_result(self):
        return SpectrumResult(
            points=self.flux_point_estimator.flux_points,
            model=self.fit.result[0].model
        )

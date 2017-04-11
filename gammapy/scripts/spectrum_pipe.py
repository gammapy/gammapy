# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..spectrum import (
    SpectrumObservationList,
    SpectrumEnergyGroupMaker,
    FluxPointEstimator,
    SpectrumResult,
)
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
    outdir : `~gammapy.extern.pathlib.Path`
        Analysis directory
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
    flux_point_estimator : `~gammapy
        Flux points estimator
        TODO: Input not used at the moment, fpe is created on the fly
    clobber : False
        Overwrite OGIP files
    stacked : Bool, optional
        Stack observations prior to fitting
    """
    def __init__(self, outdir, observations, background_estimator, extraction,
                 fit, fp_binning, flux_point_estimator=None,
                 clobber=False, stacked=False):
        self.outdir = outdir
        self.observations = observations
        self.background_estimator = background_estimator
        self.extraction = extraction
        self.fit = fit
        self.fp_binning = fp_binning
        self.flux_point_estimator = flux_point_estimator

        self.clobber = clobber
        self.stacked = stacked

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
            self._spectrum_observations = SpectrumObservationList.read(
                self.outdir / 'ogip_data')
        else:
            self.extract_all()

        spec_obs = self.extraction.observations
        if self.stacked:
            spec_obs = spec_obs.stack()
        
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
        self.fit.run(observations, outdir=self.outdir)
        egm = SpectrumEnergyGroupMaker(observations.stack())
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

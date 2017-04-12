# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
import astropy.units as u
from regions import CircleSkyRegion
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from ..data import Target
from ..background import ReflectedRegionsBackgroundEstimator
from .core import PHACountsSpectrum
from .observation import SpectrumObservation, SpectrumObservationList
from ..irf import PSF3D

__all__ = [
    'SpectrumExtraction',
]

log = logging.getLogger(__name__)


class SpectrumExtraction(object):
    """Class for creating input data to 1D spectrum fitting

    This class is responsible for extracting a
    `~gammapy.spectrum.SpectrumObservation` from a
    `~gammapy.data.DataStoreObservation`, given a certain signal extraction
    region. The background estimation is done beforehand, using e.g. the
    `~gammapy.background.ReflectedRegionsBackgroundEstimator`. For point
    sources analyzed with 'full containement' IRFs, a correction for PSF
    leakage out of the circular ON region can be applied.

    For more info see :ref:`spectral_fitting`. For a usage example see
    :gp-extra-notebook:`spectrum_analysis`

    Parameters
    ----------
    on_region : `~regions.SkyRegion`
        Signal extraction region
    e_reco : `~astropy.units.Quantity`, optional
        Reconstructed energy binning
    e_true : `~astropy.units.Quantity`, optional
        True energy binning
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    """
    DEFAULT_TRUE_ENERGY = np.logspace(-2, 2.5, 109) * u.TeV
    """True energy axis to be used if not specified otherwise"""
    DEFAULT_RECO_ENERGY = np.logspace(-2, 2, 73) * u.TeV
    """Reconstruced energy axis to be used if not specified otherwise"""

    def __init__(self, on_region, e_reco=None, e_true=None,
                 containment_correction=False):

        self.on_region = on_region
        self.e_reco = e_reco or self.DEFAULT_RECO_ENERGY
        self.e_true = e_true or self.DEFAULT_TRUE_ENERGY
        self.containment_correction = containment_correction
        if self.containment_correction and not isinstance(on_region,
                                                          CircleSkyRegion):
            raise TypeError("Incorrect region type for containment correction."
                            " Should be CircleSkyRegion.")

        self.observations = SpectrumObservationList()

    def run(self, obs_list, bkg_estimate, outdir=None, use_sherpa=False):
        """Run all steps

        Parameters
        ----------
        obs_list : `~gammapy.data.ObservationList`
            Observations to process
        bkg_estimate : `~gammapy.background.BackgroundEstimate`
            Background estimate, e.g. of
            `~gammapy.background.ReflectedRegionsBackgroundEstimator`
        outdir : Path, str
            directory to write results files to (if given)
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        """
        log.info('Running {}'.format(self))
        for obs, bkg in zip(obs_list, bkg_estimate):
            self.observations.append(self.process(obs, bkg))
        if outdir is not None:
            outdir = make_path(outdir)
            outdir.mkdir(exist_ok=True, parents=True)
            self.write(outdir, use_sherpa=use_sherpa)

    def process(self, obs, bkg):
        """Process one observation"""
        log.info('Process observation\n {}'.format(obs))
        self.make_empty_on_vector(obs, bkg)
        self.extract_counts(obs, bkg)
        self.extract_irfs(obs)
        if self.containment_correction:
            self.apply_containment_correction(obs)
        spectrum_observation = SpectrumObservation(on_vector=self._on_vector,
                                                   aeff=self._aeff,
                                                   off_vector=self._off_vector,
                                                   edisp=self._edisp)
        try:
            spectrum_observation.hi_threshold = obs.aeff.high_threshold
            spectrum_observation.lo_threshold = obs.aeff.low_threshold
        except AttributeError:
            pass
        return spectrum_observation

    def make_empty_on_vector(self, obs, bkg):
        """Create empty on vector holding all meta info"""
        log.info('Update observation meta info')
        # Copy over existing meta information
        meta = dict(obs._obs_info)
        offset = obs.pointing_radec.separation(self.on_region.center)
        log.info('Offset : {}\n'.format(offset))
        meta['OFFSET'] = offset.deg
        
        # LIVETIME is called EXPOSURE in the OGIP standard
        meta['EXPOSURE'] = meta.pop('LIVETIME')

        self._on_vector = PHACountsSpectrum(energy_lo=self.e_reco[:-1],
                                            energy_hi=self.e_reco[1:],
                                            backscal=bkg.a_on,
                                            meta=meta,)

    def extract_counts(self, obs, bkg):
        """Fill on and off vector"""
        log.info('Fill events')
        idx = self.on_region.contains(obs.events.radec)
        on_events = obs.events.select_row_subset(idx)
        self._on_vector.fill(on_events)

        # TODO: Decide which meta info to copy to off vector
        meta = dict(EXPOSURE=self._on_vector.livetime.value,
                    OBS_ID=self._on_vector.obs_id)
        self._off_vector = PHACountsSpectrum(
            backscal=bkg.a_off, is_bkg=True,
            energy_lo=self._on_vector.energy.lo,
            energy_hi=self._on_vector.energy.hi,
            meta=meta)

        self._off_vector.fill(bkg.off_events)

    def extract_irfs(self, obs):
        """Extract IRFs"""
        log.info('Extract IRFs')
        offset = self._on_vector.offset
        self._aeff = obs.aeff.to_effective_area_table(offset,
                                                      energy=self.e_true)
        self._edisp = obs.edisp.to_energy_dispersion(offset,
                                                     e_reco=self.e_reco,
                                                     e_true=self.e_true)

    def apply_containment_correction(self, obs):
        """Apply containment correction

        TODO: Split out into separate class
        """
        log.info('Apply containment correction')
        # First need psf
        angles = np.linspace(0., 1.5, 150) * u.deg
        if isinstance(obs.psf, PSF3D):
            psf = obs.psf.to_energy_dependent_table_psf(theta=self._on_vector.offset)
        else:
            psf = obs.psf.to_energy_dependent_table_psf(self._on_vector.offset, angles)

        center_energies = self._on_vector.energy.nodes
        areascal = list()
        for index, energy in enumerate(center_energies):
            try:
                correction = psf.integral(energy,
                                          0. * u.deg,
                                          self.on_region.radius)
            except:
                msg = 'Containment correction failed for bin {}, energy {}.'
                log.warn(msg.format(index, energy))
                correction = 1
            finally:
                areascal.append(correction)

        self._on_vector.areascal = areascal
        self._off_vector.areascal = areascal 

    def define_energy_threshold(self, method_lo_threshold='area_max', **kwargs):
        """Set energy threshold

        Set the high and low energy threshold for each observation based on a
        chosen method.

        Available methods for setting the low energy threshold

        * area_max : Set energy threshold at x percent of the maximum effective
                     area (x given as kwargs['percent'])

        Available methods for setting the high energy threshold

        * TBD

        Parameters
        ----------
        method_lo_threshold : {'area_max'}
            method for defining the low energy threshold
        """
        # TODO: define method for the high energy threshold

        # It is important to update the low and high threshold for ON and OFF
        # vector, otherwise Sherpa will not understand the files
        for obs in self.observations:
            if method_lo_threshold == 'area_max':
                aeff_thres = kwargs['percent'] / 100 * obs.aeff.max_area
                thres = obs.aeff.find_energy(aeff_thres)
                obs.on_vector.lo_threshold = thres
                obs.off_vector.lo_threshold = thres
            else:
                raise ValueError('Undefine method for low threshold: {}'.format(
                    method_lo_threshold))

    def write(self, outdir, ogipdir='ogip_data', use_sherpa=False):
        """Write results to disk

        Parameters
        ----------
        outdir : `~gammapy.extern.pathlib.Path`
            Output folder
        ogipdir : str, optional
            Folder name for OGIP data, default: 'ogip_data'
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        """
        log.info("Writing OGIP files to {}".format(outdir / ogipdir))
        self.observations.write(outdir / ogipdir, use_sherpa=use_sherpa)
        # TODO : add more debug plots etc. here

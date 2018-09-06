# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import astropy.units as u
from regions import CircleSkyRegion
from . import PHACountsSpectrum
from . import SpectrumObservation, SpectrumObservationList
from ..utils.scripts import make_path
from ..irf import PSF3D

__all__ = ["SpectrumExtraction"]

log = logging.getLogger(__name__)


class SpectrumExtraction(object):
    """Creating input data to 1D spectrum fitting.

    This class is responsible for extracting a
    `~gammapy.spectrum.SpectrumObservation` from a
    `~gammapy.data.DataStoreObservation`. The background estimation is done
    beforehand, using e.g. the
    `~gammapy.background.ReflectedRegionsBackgroundEstimator`. For point
    sources analyzed with 'full containment' IRFs, a correction for PSF
    leakage out of the circular ON region can be applied.

    For more info see :ref:`spectral_fitting`.

    For a usage example see :gp-extra-notebook:`spectrum_analysis`

    Parameters
    ----------
    obs_list : `~gammapy.data.ObservationList`
        Observations to process
    bkg_estimate : `~gammapy.background.BackgroundEstimate`
        Background estimate, e.g. of
        `~gammapy.background.ReflectedRegionsBackgroundEstimator`
    e_reco : `~astropy.units.Quantity`, optional
        Reconstructed energy binning
    e_true : `~astropy.units.Quantity`, optional
        True energy binning
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    max_alpha : float
        Maximum alpha value to accept, if the background was estimated using
        reflected regions this is 1 / minimum number of regions.
    use_recommended_erange : bool
        Extract spectrum only within the recommended valid energy range of the
        effective area table (default is True).
    """

    DEFAULT_TRUE_ENERGY = np.logspace(-2, 2.5, 109) * u.TeV
    """True energy axis to be used if not specified otherwise"""
    DEFAULT_RECO_ENERGY = np.logspace(-2, 2, 73) * u.TeV
    """Reconstruced energy axis to be used if not specified otherwise"""

    def __init__(
        self,
        obs_list,
        bkg_estimate,
        e_reco=None,
        e_true=None,
        containment_correction=False,
        max_alpha=1,
        use_recommended_erange=True,
    ):

        self.obs_list = obs_list
        self.bkg_estimate = bkg_estimate
        self.e_reco = e_reco if e_reco is not None else self.DEFAULT_RECO_ENERGY
        self.e_true = e_true if e_true is not None else self.DEFAULT_TRUE_ENERGY
        self.containment_correction = containment_correction
        self.max_alpha = max_alpha
        self.use_recommended_erange = use_recommended_erange
        self.observations = SpectrumObservationList()

        self.containment = None
        self._on_vector = None
        self._off_vector = None
        self._aeff = None
        self._edisp = None

    def run(self):
        """Run all steps.
        """
        log.info("Running {}".format(self))
        for obs, bkg in zip(self.obs_list, self.bkg_estimate):
            if not self._alpha_ok(obs, bkg):
                continue
            self.observations.append(self.process(obs, bkg))

    def _alpha_ok(self, obs, bkg):
        """Check if observation fulfills alpha criterion"""
        condition = bkg.a_off == 0 or bkg.a_on / bkg.a_off > self.max_alpha
        if condition:
            msg = "Skipping because {} / {} > {}"
            log.info(msg.format(bkg.a_on, bkg.a_off, self.max_alpha))
            return False
        else:
            return True

    def process(self, obs, bkg):
        """Process one observation.

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            Observation
        bkg : `~gammapy.background.BackgroundEstimate`
            Background estimate

        Returns
        -------
        spectrum_observation : `~gammapy.spectrum.SpectrumObservation`
            Spectrum observation
        """
        log.info("Process observation\n {}".format(obs))
        self.make_empty_vectors(obs, bkg)
        self.extract_counts(bkg)
        self.extract_irfs(obs)

        if self.containment_correction:
            self.apply_containment_correction(obs, bkg)
        else:
            self.containment = np.ones(self._aeff.energy.nbins)

        spectrum_observation = SpectrumObservation(
            on_vector=self._on_vector,
            aeff=self._aeff,
            off_vector=self._off_vector,
            edisp=self._edisp,
        )

        if self.use_recommended_erange:
            try:
                spectrum_observation.hi_threshold = obs.aeff.high_threshold
                spectrum_observation.lo_threshold = obs.aeff.low_threshold
            except KeyError:
                log.warning("No thresholds defined for obs {}".format(obs))

        return spectrum_observation

    def make_empty_vectors(self, obs, bkg):
        """Create empty vectors.

        This method copies over all meta info and sets up the energy binning.

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            Observation
        bkg : `~gammapy.background.BackgroundEstimate`
            Background estimate
        """
        log.info("Update observation meta info")

        offset = obs.pointing_radec.separation(bkg.on_region.center)
        log.info("Offset : {}\n".format(offset))

        self._on_vector = PHACountsSpectrum(
            energy_lo=self.e_reco[:-1],
            energy_hi=self.e_reco[1:],
            backscal=bkg.a_on,
            offset=offset,
            livetime=obs.observation_live_time_duration,
            obs_id=obs.obs_id,
        )

        self._off_vector = self._on_vector.copy()
        self._off_vector.is_bkg = True
        self._off_vector.backscal = bkg.a_off

    def extract_counts(self, bkg):
        """Fill on and off vector for one observation.

        Parameters
        ----------
        bkg : `~gammapy.background.BackgroundEstimate`
            Background estimate
        """
        log.info("Fill events")
        self._on_vector.fill(bkg.on_events)
        self._off_vector.fill(bkg.off_events)

    def extract_irfs(self, obs):
        """Extract IRFs.

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            Observation
        """
        log.info("Extract IRFs")
        offset = self._on_vector.offset
        self._aeff = obs.aeff.to_effective_area_table(offset, energy=self.e_true)
        self._edisp = obs.edisp.to_energy_dispersion(
            offset, e_reco=self.e_reco, e_true=self.e_true
        )

    def apply_containment_correction(self, obs, bkg):
        """Apply PSF containment correction.

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            observation
        bkg : `~gammapy.background.BackgroundEstimate`
            background esimate
        """
        # TODO: This should be split out into a separate class
        if not isinstance(bkg.on_region, CircleSkyRegion):
            raise TypeError(
                "Incorrect region type for containment correction."
                " Should be CircleSkyRegion."
            )

        log.info("Apply containment correction")
        # First need psf
        angles = np.linspace(0., 1.5, 150) * u.deg
        offset = self._on_vector.offset
        if isinstance(obs.psf, PSF3D):
            psf = obs.psf.to_energy_dependent_table_psf(theta=offset)
        else:
            psf = obs.psf.to_energy_dependent_table_psf(offset, angles)

        center_energies = self._aeff.energy.nodes
        containment = []
        for index, energy in enumerate(center_energies):
            try:
                cont_ = psf.integral(energy, 0. * u.deg, bkg.on_region.radius)
            except:
                msg = "Containment correction failed for bin {}, energy {}."
                log.warning(msg.format(index, energy))
                cont_ = 1
            finally:
                containment.append(cont_)

        self.containment = np.array(containment)
        self._aeff.data.data *= self.containment

    def compute_energy_threshold(self, **kwargs):
        """Compute and set the safe energy threshold for all observations.

        See `SpectrumObservation.compute_energy_threshold` for full
        documentation about the options.
        """

        for obs in self.observations:
            obs.compute_energy_threshold(**kwargs)

    def write(self, outdir, ogipdir="ogip_data", use_sherpa=False, overwrite=False):
        """Write results to disk.

        Parameters
        ----------
        outdir : `~gammapy.extern.pathlib.Path`
            Output folder
        ogipdir : str, optional
            Folder name for OGIP data, default: 'ogip_data'
        use_sherpa : bool, optional
            Write Sherpa compliant files?
        overwrite : bool
            Overwrite existing files?
        """
        outdir = make_path(outdir)
        log.info("Writing OGIP files to {}".format(outdir / ogipdir))
        outdir.mkdir(exist_ok=True, parents=True)
        self.observations.write(
            outdir / ogipdir, use_sherpa=use_sherpa, overwrite=overwrite
        )

        # TODO : add more debug plots etc. here

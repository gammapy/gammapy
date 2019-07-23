# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from pathlib import Path
import numpy as np
import astropy.units as u
from regions import CircleSkyRegion
from ..utils.scripts import make_path
from ..irf import PSF3D, apply_containment_fraction, compute_energy_thresholds
from .core import CountsSpectrum
from .dataset import SpectrumDatasetOnOff

__all__ = ["SpectrumExtraction"]

log = logging.getLogger(__name__)


class SpectrumExtraction:
    """Creating input data to 1D spectrum fitting.

    This class is responsible for extracting a
    `~gammapy.spectrum.SpectrumObservation` from a
    `~gammapy.data.DataStoreObservation`. The background estimation is done
    beforehand, using e.g. the
    `~gammapy.background.ReflectedRegionsBackgroundEstimator`. For point
    sources analyzed with 'full containment' IRFs, a correction for PSF
    leakage out of the circular ON region can be applied.

    For more info see :ref:`spectral_fitting`.

    For a usage example see :gp-notebook:`spectrum_analysis`

    Parameters
    ----------
    observations : `~gammapy.data.Observations`
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
        observations,
        bkg_estimate,
        e_reco=None,
        e_true=None,
        containment_correction=False,
        max_alpha=1,
        use_recommended_erange=True,
    ):

        self.observations = observations
        self.bkg_estimate = bkg_estimate
        self.e_reco = e_reco if e_reco is not None else self.DEFAULT_RECO_ENERGY
        self.e_true = e_true if e_true is not None else self.DEFAULT_TRUE_ENERGY
        self.containment_correction = containment_correction
        self.max_alpha = max_alpha
        self.use_recommended_erange = use_recommended_erange
        self.spectrum_observations = []

        self.containment = None
        self._on_vector = None
        self._off_vector = None
        self._aeff = None
        self._edisp = None

    def run(self):
        """Run all steps.
        """
        log.info("Running {}".format(self))
        for obs, bkg in zip(self.observations, self.bkg_estimate):
            if not self._alpha_ok(bkg):
                continue
            self.spectrum_observations.append(self.process(obs, bkg))

    def _alpha_ok(self, bkg):
        """Check if observation fulfills alpha criterion."""
        condition = bkg.a_off == 0 or bkg.a_on / bkg.a_off > self.max_alpha
        if condition:
            msg = "Skipping because {} / {} > {}"
            log.info(msg.format(bkg.a_on, bkg.a_off, self.max_alpha))
            return False
        else:
            return True

    def process(self, observation, bkg):
        """Process one observation.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            Observation
        bkg : `~gammapy.background.BackgroundEstimate`
            Background estimate

        Returns
        -------
        spectrum_observation : `~gammapy.spectrum.SpectrumObservation`
            Spectrum observation
        """
        log.info("Process observation\n {}".format(observation))
        self.make_empty_vectors(observation, bkg)
        self.extract_counts(bkg)
        self.extract_irfs(observation, bkg)

        if self.containment_correction:
            self.apply_containment_correction(observation, bkg)
        else:
            self.containment = np.ones(self._aeff.energy.nbin)

        spectrum_observation = SpectrumDatasetOnOff(
            counts=self._on_vector,
            aeff=self._aeff,
            counts_off=self._off_vector,
            edisp=self._edisp,
            livetime=observation.observation_live_time_duration,
            acceptance=1,
            acceptance_off=bkg.a_off,
            obs_id=observation.obs_id,
            gti=observation.gti,
        )

        if self.use_recommended_erange:
            try:
                e_max = observation.aeff.high_threshold
                e_min = observation.aeff.low_threshold
                spectrum_observation.mask_safe = spectrum_observation.counts.energy_mask(
                    emin=e_min, emax=e_max
                )
            except KeyError:
                log.warning("No thresholds defined for obs {}".format(observation))

        return spectrum_observation

    def make_empty_vectors(self, observation, bkg):
        """Create empty vectors.

        This method copies over all meta info and sets up the energy binning.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            Observation
        bkg : `~gammapy.background.BackgroundEstimate`
            Background estimate
        """
        log.info("Update observation meta info")

        offset = observation.pointing_radec.separation(bkg.on_region.center)
        log.info("Offset : {}\n".format(offset))

        self._on_vector = CountsSpectrum(
            energy_lo=self.e_reco[:-1], energy_hi=self.e_reco[1:]
        )

        self._off_vector = self._on_vector.copy()

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

    def extract_irfs(self, observation, bkg):
        """Extract IRFs.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            Observation
        """
        log.info("Extract IRFs")
        offset = observation.pointing_radec.separation(bkg.on_region.center)
        self._aeff = observation.aeff.to_effective_area_table(
            offset, energy=self.e_true
        )
        self._edisp = observation.edisp.to_energy_dispersion(
            offset, e_reco=self.e_reco, e_true=self.e_true
        )

    def apply_containment_correction(self, observation, bkg):
        """Apply PSF containment correction.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            observation
        bkg : `~gammapy.background.BackgroundEstimate`
            background esimate
        """
        if not isinstance(bkg.on_region, CircleSkyRegion):
            raise TypeError(
                "Incorrect region type for containment correction."
                " Should be CircleSkyRegion."
            )

        log.info("Apply containment correction")
        # First need psf
        angles = np.linspace(0.0, 1.5, 150) * u.deg
        offset = observation.pointing_radec.separation(bkg.on_region.center)
        if isinstance(observation.psf, PSF3D):
            psf = observation.psf.to_energy_dependent_table_psf(theta=offset)
        else:
            psf = observation.psf.to_energy_dependent_table_psf(offset, angles)

        new_aeff = apply_containment_fraction(self._aeff, psf, bkg.on_region.radius)

        # TODO: check whether keeping containment is necessary
        self.containment = new_aeff.data.data.value / self._aeff.data.data.value
        self._aeff = new_aeff

    def compute_energy_threshold(self, **kwargs):
        """Compute and set the safe energy threshold for all observations.

        See `~gammapy.irf.compute_energy_thresholds` for full
        documentation about the options.
        """
        for obs in self.spectrum_observations:
            emin, emax = compute_energy_thresholds(obs.aeff, obs.edisp, **kwargs)
            mask_safe = obs.counts.energy_mask(emin=emin, emax=emax)

            if obs.mask_safe is not None:
                obs.mask_safe &= mask_safe
            else:
                obs.mask_safe = mask_safe

    def write(self, outdir, ogipdir="ogip_data", use_sherpa=False, overwrite=False):
        """Write results to disk as OGIP format.

        Parameters
        ----------
        outdir : `pathlib.Path`
            Output folder
        ogipdir : str, optional
            Folder name for OGIP data, default: 'ogip_data'
        use_sherpa : bool, optional
            Write Sherpa compliant files?
        overwrite : bool
            Overwrite existing files?
        """
        outdir = Path.cwd() if outdir is None else Path(outdir)
        outdir = make_path(outdir / ogipdir)
        log.info("Writing OGIP files to {}".format(outdir))
        outdir.mkdir(exist_ok=True, parents=True)
        for obs in self.spectrum_observations:
            obs.to_ogip_files(outdir=outdir, use_sherpa=use_sherpa, overwrite=overwrite)
        # TODO : add more debug plots etc. here

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from regions import CircleSkyRegion
from ..irf import PSF3D, apply_containment_fraction, compute_energy_thresholds
from .core import CountsSpectrum
from .dataset import SpectrumDataset
from ..background import ReflectedRegionsFinder

__all__ = ["SpectrumDatasetMakerObs"]

class SpectrumDatasetMakerObs:
    """Creates and fill a `~gammapy.spectrum.SpectrumDataset` for a single observation.

    Counts are extracted from an on region using a tangent projection centered on
    the pointing position.

    Reduced IRFs are extracted at the region center.
    TODO: allow for extended source extraction and average over region

    Parameters
    ----------
    observation : `~gammapy.data.DataStoreObservation`
        the observation
    on_region : `~regions.SkyRegion`
        the on region
    e_reco : `~astropy.units.Quantity`, optional
        Reconstructed energy binning
    e_true : `~astropy.units.Quantity`, optional
        True energy binning
   containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    use_recommended_erange : bool
        Extract spectrum only within the recommended valid energy range of the
        effective area table (default is True).

    Returns
    -------
    dataset : `~gammapy.spectrum.SpectrumDataset`
        the output dataset
    """
    DEFAULT_TRUE_ENERGY = np.logspace(-2, 2.5, 109) * u.TeV
    """True energy axis to be used if not specified otherwise"""
    DEFAULT_RECO_ENERGY = np.logspace(-2, 2, 73) * u.TeV
    """Reconstruced energy axis to be used if not specified otherwise"""

    def __init__(self,
        observation,
        on_region,
        e_reco=None,
        e_true=None,
        containment_correction=False,
        use_recommended_erange=True,
    ):
        self.observation = observation
        self.on_region = on_region
        self.e_reco = e_reco if e_reco is not None else self.DEFAULT_RECO_ENERGY
        self.e_true = e_true if e_true is not None else self.DEFAULT_TRUE_ENERGY
        self.containment_correction = containment_correction
        self.use_recommended_erange = use_recommended_erange

        self.dataset = self.make_dataset()
        self.reference_map = ReflectedRegionsFinder.make_reference_map(self.on_region,
                                                                      self.observation.pointing_radec,
                                                                      binsz=0.01*u.deg)
    def make_dataset(self):
        """Create empty vector.
        This method copies over all meta info and sets up the energy binning.
        """
        on_vector = CountsSpectrum(
            energy_lo=self.e_reco[:-1], energy_hi=self.e_reco[1:]
        )

#        on_vector.meta = self.observation.meta
        return SpectrumDataset(counts=on_vector, livetime=self.observation.observation_live_time_duration)

    def extract_counts(self):
        on_events = self.observation.events.select_region(self.on_region, self.reference_map.geom.wcs)
        self.dataset.counts.fill(on_events)

    def extract_irfs(self):
        """Extract IRFs."""
        offset = self.observation.pointing_radec.separation(self.on_region.center)
        self.dataset.aeff = self.observation.aeff.to_effective_area_table(
            offset, energy=self.e_true
        )
        self.dataset.edisp = self.observation.edisp.to_energy_dispersion(
            offset, e_reco=self.e_reco, e_true=self.e_true
        )

    def extract_bkg(self):
        """Extract background from IRF"""
        offset = self.observation.pointing_radec.separation(self.on_region.center)
        bkg_data = self.observation.bkg.evaluate_integrate(
            fov_lon=0 * u.deg, fov_lat=offset, energy_reco=self.e_reco
        )
        self.dataset.background = CountsSpectrum(
            energy_lo=self.e_reco[:-1], energy_hi=self.e_reco[1:], data=bkg_data
        )

    def run(self):
        """Process the observation."""

        self.extract_counts()
        self.extract_irfs()

        try:
            self.extract_bkg()
        except:
            self.dataset.background = None

        if self.containment_correction:
            self.apply_containment_correction()

        if self.use_recommended_erange:
            try:
                e_max = self.observation.aeff.high_threshold
                e_min = self.observation.aeff.low_threshold
                self.dataset.mask_safe = self.dataset.counts.energy_mask(
                    emin=e_min, emax=e_max
                )
            except KeyError:
                # TODO : use log
                print("No thresholds defined for obs {}".format(self.observation))

        return self.dataset

    def apply_containment_correction(self):
        """Apply PSF containment correction.       """
        if not isinstance(self.on_region, CircleSkyRegion):
            raise TypeError(
                "Incorrect region type for containment correction."
                " Should be CircleSkyRegion."
            )

        # First need psf
        angles = np.linspace(0.0, 1.5, 150) * u.deg
        offset = self.observation.pointing_radec.separation(self.on_region.center)
        if isinstance(self.observation.psf, PSF3D):
            psf = self.observation.psf.to_energy_dependent_table_psf(theta=offset)
        else:
            psf = self.observation.psf.to_energy_dependent_table_psf(offset, angles)

        new_aeff = apply_containment_fraction(self.dataset.aeff, psf, self.on_region.radius)

        self.dataset.aeff = new_aeff

    def compute_energy_threshold(self, **kwargs):
        """Compute and set the safe energy threshold for all observations.

        See `~gammapy.irf.compute_energy_thresholds` for full
        documentation about the options.
        """
        emin, emax = compute_energy_thresholds(self.dataset.aeff, self.dataset.edisp, **kwargs)
        mask_safe = self.dataset.counts.energy_mask(emin=emin, emax=emax)

        if self.dataset.mask_safe is not None:
            self.dataset.mask_safe &= mask_safe
        else:
            self.dataset.mask_safe = mask_safe

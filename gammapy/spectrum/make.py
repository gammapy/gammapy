import logging
from astropy import units as u
from astropy.coordinates import Angle
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
from gammapy.irf import EnergyDependentMultiGaussPSF, apply_containment_fraction
from gammapy.maps import WcsGeom
from gammapy.maps.geom import frame_to_coordsys
from .core import CountsSpectrum
from .dataset import SpectrumDataset

log = logging.getLogger(__name__)


class SpectrumDatasetMaker:
    """Make spectrum for a single IACT observation.

    The irfs and background are computed at a single fixed offset,
    which is recommend only for point-sources.

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region to compute spectrum dataset for.
    e_reco : `~astropy.units.Quantity`
        Reconstructed energy binning
    e_true : `~astropy.units.Quantity`
        True energy binning
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    binsz : `~astropy.coordinates.Angle`
        Reference map bin size.
    width : `~astropy.coordinates.Angle`
        Reference map width, should encompass the whole region.

    """

    def __init__(
        self,
        region,
        e_reco,
        e_true=None,
        containment_correction=True,
        binsz="0.01 deg",
        width="0.5 deg",
    ):
        self.region = region
        self.e_reco = e_reco
        self.e_true = e_true or e_reco
        self.containment_correction = containment_correction
        self.binsz = Angle(binsz)
        self.width = Angle(width)

    # TODO: move this to a RegionGeom class
    @lazyproperty
    def geom_ref(self):
        """Reference geometry to project region"""
        coordsys = frame_to_coordsys(self.region.center.frame.name)
        return WcsGeom.create(
            skydir=self.region.center,
            width=self.width,
            binsz=self.binsz,
            proj="TAN",
            coordsys=coordsys,
        )

    @lazyproperty
    # TODO: move this to a RegionGeom class
    def region_solid_angle(self):
        """Solid angle of the region"""
        geom = self.geom_ref
        coords = geom.get_coord()
        solid_angle = geom.solid_angle()
        mask = self.region.contains(coords.skycoord, wcs=geom.wcs)
        return solid_angle[mask].sum()

    def make_counts(self, observation):
        """Make counts

        Parameters
        ----------
        observation: `DataStoreObservation`
            Observation to compute effective area for.

        Returns
        -------
        counts : `CountsSpectrum`
            Counts spectrum
        """
        energy_hi = self.e_reco[1:]
        energy_lo = self.e_reco[:-1]

        counts = CountsSpectrum(energy_hi=energy_hi, energy_lo=energy_lo)
        events_region = observation.events.select_region(
            self.region, wcs=self.geom_ref.wcs
        )
        counts.fill(events_region)
        return counts

    def make_background(self, observation):
        """Make background

        Parameters
        ----------
        observation: `DataStoreObservation`
            Observation to compute effective area for.

        Returns
        -------
        background : `CountsSpectrum`
            Background spectrum
        """
        offset = observation.pointing_radec.separation(self.region.center)
        energy_hi = self.e_reco[1:]
        energy_lo = self.e_reco[:-1]

        bkg = observation.bkg

        data = bkg.evaluate_integrate(
            fov_lon=0 * u.deg, fov_lat=offset, energy_reco=self.e_reco
        )

        data *= self.region_solid_angle
        data *= observation.observation_time_duration

        counts = CountsSpectrum(
            energy_hi=energy_hi, energy_lo=energy_lo, data=data.to_value(""), unit=""
        )
        return counts

    def make_aeff(self, observation):
        """Make effective area

        Parameters
        ----------
        observation: `DataStoreObservation`
            Observation to compute effective area for.

        Returns
        -------
        aeff : `EffectiveAreaTable`
            Effective area table.
        """
        offset = observation.pointing_radec.separation(self.region.center)
        aeff = observation.aeff.to_effective_area_table(offset, energy=self.e_true)

        if self.containment_correction:
            if not isinstance(self.region, CircleSkyRegion):
                raise TypeError(
                    "Containment correction only support for circular regions."
                )
            psf = observation.psf

            if isinstance(psf, EnergyDependentMultiGaussPSF):
                psf = psf.to_psf3d()

            table_psf = psf.to_energy_dependent_table_psf(theta=offset)
            aeff = apply_containment_fraction(aeff, table_psf, self.region.radius)

        return aeff

    def make_edisp(self, observation):
        """Make energy dispersion

        Parameters
        ----------
        observation: `DataStoreObservation`
            Observation to compute edisp for.

        Returns
        -------
        edisp : `EnergyDispersion`
            Energy dispersion

        """
        offset = observation.pointing_radec.separation(self.region.center)
        edisp = observation.edisp.to_energy_dispersion(
            offset, e_reco=self.e_reco, e_true=self.e_true
        )
        return edisp

    def run(self, observation, selection=None):
        """Make spectrum dataset.

        Parameters
        ----------
        observation: `DataStoreObservation`
            Observation to reduce.
        selection : list
            List of str, selecting which maps to make.
            Available: 'counts', 'aeff', 'background', 'edisp'
            By default, all spectra are made.

        Returns
        -------
        dataset : `SpectrumDataset`
            Spectrum dataset.
        """
        if selection is None:
            selection = ["counts", "background", "aeff", "edisp"]

        kwargs = {}

        kwargs["gti"] = observation.gti
        kwargs["name"] = "obs_{}".format(observation.obs_id)
        kwargs["livetime"] = observation.observation_live_time_duration

        if "counts" in selection:
            kwargs["counts"] = self.make_counts(observation)

        if "background" in selection:
            kwargs["background"] = self.make_background(observation)

        if "aeff" in selection:
            kwargs["aeff"] = self.make_aeff(observation)

        if "edisp" in selection:
            kwargs["edisp"] = self.make_edisp(observation)

        return SpectrumDataset(**kwargs)

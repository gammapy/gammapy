# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
from gammapy.maps import WcsGeom
from gammapy.maps.geom import frame_to_coordsys
from .core import CountsSpectrum
from .dataset import SpectrumDataset

__all__ = ["SpectrumDatasetMaker"]

log = logging.getLogger(__name__)

__all__ = ["SpectrumDatasetMaker"]


class SpectrumDatasetMaker:
    """Make spectrum for a single IACT observation.

    The irfs and background are computed at a single fixed offset,
    which is recommend only for point-sources.

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region to compute spectrum dataset for.
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    selection : list
        List of str, selecting which maps to make.
        Available: 'counts', 'aeff', 'background', 'edisp'
        By default, all spectra are made.

    """
    available_selection = ["counts", "background", "aeff", "edisp"]

    def __init__(self, region, containment_correction=False, selection=None):
        self.region = region
        self.containment_correction = containment_correction

        if selection is None:
            selection = self.available_selection

        self.selection = selection

    # TODO: move this to a RegionGeom class
    @lazyproperty
    def geom_ref(self):
        """Reference geometry to project region"""
        coordsys = frame_to_coordsys(self.region.center.frame.name)
        return WcsGeom.create(
            skydir=self.region.center,
            npix=(1, 1),
            binsz=1,
            proj="TAN",
            coordsys=coordsys,
        )

    def make_counts(self, energy_axis, observation):
        """Make counts

        Parameters
        ----------
        energy_axis : `MapAxis`
            Reconstructed energy axis.
        observation: `DataStoreObservation`
            Observation to compute effective area for.

        Returns
        -------
        counts : `CountsSpectrum`
            Counts spectrum
        """
        edges = energy_axis.edges

        counts = CountsSpectrum(
            energy_hi=edges[1:], energy_lo=edges[:-1], region=self.region
        )
        events_region = observation.events.select_region(
            self.region, wcs=self.geom_ref.wcs
        )
        counts.fill_events(events_region)
        return counts

    def make_background(self, energy_axis, observation):
        """Make background

        Parameters
        ----------
        energy_axis : `MapAxis`
            Reconstructed energy axis.
        observation: `DataStoreObservation`
            Observation to compute effective area for.

        Returns
        -------
        background : `CountsSpectrum`
            Background spectrum
        """
        if not isinstance(self.region, CircleSkyRegion):
            raise TypeError(
                "Background computation only supported for circular regions."
            )

        offset = observation.pointing_radec.separation(self.region.center)
        e_reco = energy_axis.edges

        bkg = observation.bkg

        data = bkg.evaluate_integrate(
            fov_lon=0 * u.deg, fov_lat=offset, energy_reco=e_reco
        )

        solid_angle = 2 * np.pi * (1 - np.cos(self.region.radius)) * u.sr
        data *= solid_angle
        data *= observation.observation_time_duration

        return CountsSpectrum(
            energy_hi=e_reco[1:], energy_lo=e_reco[:-1], data=data.to_value(""), unit=""
        )

    def make_aeff(self, energy_axis_true, observation):
        """Make effective area

        Parameters
        ----------
        energy_axis_true : `MapAxis`
            True energy axis.
        observation: `DataStoreObservation`
            Observation to compute effective area for.

        Returns
        -------
        aeff : `EffectiveAreaTable`
            Effective area table.
        """
        offset = observation.pointing_radec.separation(self.region.center)
        aeff = observation.aeff.to_effective_area_table(offset, energy=energy_axis_true.edges)

        if self.containment_correction:
            if not isinstance(self.region, CircleSkyRegion):
                raise TypeError(
                    "Containment correction only supported for circular regions."
                )
            psf = observation.psf.to_energy_dependent_table_psf(theta=offset)
            containment = psf.containment(aeff.energy.center, self.region.radius)
            aeff.data.data *= containment.squeeze()

        return aeff

    def make_edisp(self, energy_axis, energy_axis_true, observation):
        """Make energy dispersion

        Parameters
        ----------
        energy_axis : `MapAxis`
            Reconstructed energy axis.
        energy_axis_true : `MapAxis`
            True energy axis.
        observation: `DataStoreObservation`
            Observation to compute edisp for.

        Returns
        -------
        edisp : `EnergyDispersion`
            Energy dispersion

        """
        offset = observation.pointing_radec.separation(self.region.center)
        edisp = observation.edisp.to_energy_dispersion(
            offset, e_reco=energy_axis.edges, e_true=energy_axis_true.edges
        )
        return edisp

    def run(self, dataset, observation):
        """Make spectrum dataset.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset.
        observation: `DataStoreObservation`
            Observation to reduce.

        Returns
        -------
        dataset : `SpectrumDataset`
            Spectrum dataset.
        """

        kwargs = {
            "name": f"{observation.obs_id}",
            "gti": observation.gti,
            "livetime": observation.observation_live_time_duration,
        }
        energy_axis = dataset.counts.energy
        energy_axis_true = dataset.aeff.data.axis("energy")

        if "counts" in self.selection:
            kwargs["counts"] = self.make_counts(energy_axis, observation)

        if "background" in self.selection:
            kwargs["background"] = self.make_background(energy_axis, observation)

        if "aeff" in self.selection:
            kwargs["aeff"] = self.make_aeff(energy_axis_true, observation)

        if "edisp" in self.selection:

            kwargs["edisp"] = self.make_edisp(energy_axis, energy_axis_true, observation)

        return SpectrumDataset(**kwargs)

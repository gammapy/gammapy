# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from regions import CircleSkyRegion
from gammapy.maps import WcsGeom
from .core import CountsSpectrum
from .dataset import SpectrumDataset

__all__ = ["SpectrumDatasetMaker"]

log = logging.getLogger(__name__)


class SpectrumDatasetMaker:
    """Make spectrum for a single IACT observation.

    The irfs and background are computed at a single fixed offset,
    which is recommend only for point-sources.

    Parameters
    ----------
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    selection : list
        List of str, selecting which maps to make.
        Available: 'counts', 'aeff', 'background', 'edisp'
        By default, all spectra are made.
    """

    available_selection = ["counts", "background", "aeff", "edisp"]

    def __init__(self, containment_correction=False, selection=None):
        self.containment_correction = containment_correction

        if selection is None:
            selection = self.available_selection

        self.selection = selection

    # TODO: move this to a RegionGeom class
    @staticmethod
    def geom_ref(region):
        """Reference geometry to project region"""
        frame = region.center.frame.name
        return WcsGeom.create(
            skydir=region.center, npix=(1, 1), binsz=1, proj="TAN", frame=frame
        )

    def make_counts(self, region, energy_axis, observation):
        """Make counts.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to compute counts spectrum for.
        energy_axis : `~gammapy.maps.MapAxis`
            Reconstructed energy axis.
        observation: `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        counts : `~gammapy.spectrum.CountsSpectrum`
            Counts spectrum
        """
        edges = energy_axis.edges

        counts = CountsSpectrum(
            energy_hi=edges[1:], energy_lo=edges[:-1], region=region, wcs=self.geom_ref(region).wcs
        )
        events_region = observation.events.select_region(
            region, wcs=self.geom_ref(region).wcs
        )
        counts.fill_events(events_region)
        return counts

    @staticmethod
    def make_background(region, energy_axis, observation):
        """Make background.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to compute background spectrum for.
        energy_axis : `~gammapy.maps.MapAxis`
            Reconstructed energy axis.
        observation: `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        background : `~gammapy.spectrum.CountsSpectrum`
            Background spectrum
        """
        if not isinstance(region, CircleSkyRegion):
            raise TypeError(
                "Background computation only supported for circular regions."
            )

        offset = observation.pointing_radec.separation(region.center)
        e_reco = energy_axis.edges

        bkg = observation.bkg

        data = bkg.evaluate_integrate(
            fov_lon=0 * u.deg, fov_lat=offset, energy_reco=e_reco
        )

        solid_angle = 2 * np.pi * (1 - np.cos(region.radius)) * u.sr
        data *= solid_angle
        data *= observation.observation_time_duration

        return CountsSpectrum(
            energy_hi=e_reco[1:], energy_lo=e_reco[:-1], data=data.to_value(""), unit="",
        )

    def make_aeff(self, region, energy_axis_true, observation):
        """Make effective area.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to compute background effective area.
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis.
        observation: `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        aeff : `~gammapy.irf.EffectiveAreaTable`
            Effective area table.
        """
        offset = observation.pointing_radec.separation(region.center)
        aeff = observation.aeff.to_effective_area_table(
            offset, energy=energy_axis_true.edges
        )

        if self.containment_correction:
            if not isinstance(region, CircleSkyRegion):
                raise TypeError(
                    "Containment correction only supported for circular regions."
                )
            psf = observation.psf.to_energy_dependent_table_psf(theta=offset)
            containment = psf.containment(aeff.energy.center, region.radius)
            aeff.data.data *= containment.squeeze()

        return aeff

    @staticmethod
    def make_edisp(position, energy_axis, energy_axis_true, observation):
        """Make energy dispersion.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position to compute energy dispersion for.
        energy_axis : `~gammapy.maps.MapAxis`
            Reconstructed energy axis.
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis.
        observation: `~gammapy.data.Observation`
            Observation to compute edisp for.

        Returns
        -------
        edisp : `~gammapy.irf.EDispKernel`
            Energy dispersion
        """
        offset = observation.pointing_radec.separation(position)
        return observation.edisp.to_energy_dispersion(
            offset, e_reco=energy_axis.edges, e_true=energy_axis_true.edges
        )

    def run(self, dataset, observation):
        """Make spectrum dataset.

        Parameters
        ----------
        dataset : `~gammapy.spectrum.SpectrumDataset`
            Spectrum dataset.
        observation: `~gammapy.data.Observation`
            Observation to reduce.

        Returns
        -------
        dataset : `~gammapy.spectrum.SpectrumDataset`
            Spectrum dataset.
        """
        kwargs = {
            "gti": observation.gti,
            "livetime": observation.observation_live_time_duration,
        }
        energy_axis = dataset.counts.energy
        energy_axis_true = dataset.aeff.data.axis("energy_true")
        region = dataset.counts.region

        if "counts" in self.selection:
            kwargs["counts"] = self.make_counts(region, energy_axis, observation)

        if "background" in self.selection:
            kwargs["background"] = self.make_background(
                region, energy_axis, observation
            )

        if "aeff" in self.selection:
            kwargs["aeff"] = self.make_aeff(region, energy_axis_true, observation)

        if "edisp" in self.selection:

            kwargs["edisp"] = self.make_edisp(
                region.center, energy_axis, energy_axis_true, observation
            )

        return SpectrumDataset(name=dataset.name, **kwargs)

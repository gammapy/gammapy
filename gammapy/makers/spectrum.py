# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.datasets import SpectrumDataset
from gammapy.irf import EDispKernelMap
from gammapy.maps import RegionNDMap
from .core import Maker

__all__ = ["SpectrumDatasetMaker"]

log = logging.getLogger(__name__)


class SpectrumDatasetMaker(Maker):
    """Make spectrum for a single IACT observation.

    The irfs and background are computed at a single fixed offset,
    which is recommend only for point-sources.

    Parameters
    ----------
    selection : list
        List of str, selecting which maps to make.
        Available: 'counts', 'exposure', 'background', 'edisp'
        By default, all spectra are made.
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    """

    tag = "SpectrumDatasetMaker"
    available_selection = ["counts", "background", "exposure", "edisp"]

    def __init__(self, selection=None, containment_correction=False):
        self.containment_correction = containment_correction

        if selection is None:
            selection = self.available_selection

        self.selection = selection

    @staticmethod
    def make_counts(geom, observation):
        """Make counts map.

        Parameters
        ----------
        geom : `~gammapy.maps.RegionGeom`
            Reference map geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        counts : `~gammapy.maps.RegionNDMap`
            Counts map.
        """
        counts = RegionNDMap.from_geom(geom)
        counts.fill_events(observation.events)
        return counts

    @staticmethod
    def make_background(geom, observation):
        """Make background.

        Parameters
        ----------
        geom : `~gammapy.maps.RegionGeom`
            Reference map geom.
        observation: `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        background : `~gammapy.maps.RegionNDMap`
            Background spectrum
        """
        offset = observation.pointing_radec.separation(geom.center_skydir)
        e_reco = geom.axes["energy"].edges

        bkg = observation.bkg

        data = bkg.evaluate_integrate(
            fov_lon=0 * u.deg, fov_lat=offset, energy_reco=e_reco
        )

        data *= geom.solid_angle()
        data *= observation.observation_time_duration
        return RegionNDMap.from_geom(geom=geom, data=data.to_value(""))

    def make_exposure(self, geom, observation):
        """Make exposure.

        Parameters
        ----------
        geom : `~gammapy.maps.RegionGeom`
            Reference map geom.
        observation: `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        exposure : `~gammapy.irf.EffectiveAreaTable`
            Exposure map.
        """
        offset = observation.pointing_radec.separation(geom.center_skydir)
        energy = geom.axes["energy_true"]

        data = observation.aeff.data.evaluate(offset=offset, energy_true=energy.center)

        if self.containment_correction:
            if not isinstance(geom.region, CircleSkyRegion):
                raise TypeError(
                    "Containment correction only supported for circular regions."
                )
            psf = observation.psf.to_energy_dependent_table_psf(theta=offset)
            containment = psf.containment(energy.center, geom.region.radius)
            data *= containment.squeeze()

        data = data * observation.observation_live_time_duration
        meta = {"livetime": observation.observation_live_time_duration}
        return RegionNDMap.from_geom(geom, data=data.value, unit=data.unit, meta=meta)

    def make_edisp_kernel(self, geom, observation):
        """Make energy dispersion.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom. Must contain "energy" and "energy_true" axes in that order.
        observation: `~gammapy.data.Observation`
            Observation to compute edisp for.

        Returns
        -------
        edisp : `~gammapy.irf.EDispKernelMap`
            Energy dispersion kernel map
        """
        position = geom.center_skydir
        energy_axis = geom.axes["energy"]
        energy_axis_true = geom.axes["energy_true"]

        offset = observation.pointing_radec.separation(position)

        kernel = observation.edisp.to_edisp_kernel(
            offset, energy=energy_axis.edges, energy_true=energy_axis_true.edges
        )

        edisp = EDispKernelMap.from_edisp_kernel(kernel, geom=geom.to_image())

        exposure = self.make_exposure(geom.squash("energy"), observation)
        edisp.exposure_map.data = exposure.data[:, :, np.newaxis, :]
        return edisp

    @staticmethod
    def make_meta_table(observation):
        """Make info meta table.

        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation

        Returns
        -------
        meta_table: `~astropy.table.Table`
        """
        meta_table = Table()
        meta_table["TELESCOP"] = [observation.aeff.meta.get("TELESCOP")]
        meta_table["OBS_ID"] = [observation.obs_id]
        meta_table["RA_PNT"] = [observation.pointing_radec.icrs.ra.deg] * u.deg
        meta_table["DEC_PNT"] = [observation.pointing_radec.icrs.dec.deg] * u.deg
        return meta_table

    def run(self, dataset, observation):
        """Make spectrum dataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.SpectrumDataset`
            Spectrum dataset.
        observation: `~gammapy.data.Observation`
            Observation to reduce.

        Returns
        -------
        dataset : `~gammapy.datasets.SpectrumDataset`
            Spectrum dataset.
        """
        kwargs = {
            "gti": observation.gti,
            "meta_table": self.make_meta_table(observation),
        }

        if "counts" in self.selection:
            kwargs["counts"] = self.make_counts(dataset.counts.geom, observation)

        if "background" in self.selection:
            kwargs["background"] = self.make_background(
                dataset.counts.geom, observation
            )

        if "exposure" in self.selection:
            kwargs["exposure"] = self.make_exposure(dataset.exposure.geom, observation)

        if "edisp" in self.selection:
            kwargs["edisp"] = self.make_edisp_kernel(
                dataset.edisp.edisp_map.geom, observation
            )

        return SpectrumDataset(name=dataset.name, **kwargs)

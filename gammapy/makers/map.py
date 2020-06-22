# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from astropy.table import Table
import astropy.units as u
from gammapy.datasets import MapDataset
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.maps import Map
from gammapy.modeling.models import BackgroundModel
from .utils import (
    make_edisp_map,
    make_edisp_kernel_map,
    make_map_background_irf,
    make_map_exposure_true_energy,
    make_psf_map,
)

__all__ = ["MapDatasetMaker"]

log = logging.getLogger(__name__)


class MapDatasetMaker:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    background_oversampling : int
        Background evaluation oversampling factor in energy.
    selection : list
        List of str, selecting which maps to make.
        Available: 'counts', 'exposure', 'background', 'psf', 'edisp'
        By default, all maps are made.
    """

    available_selection = ["counts", "exposure", "background", "psf", "edisp"]

    def __init__(self, background_oversampling=None, selection=None):
        self.background_oversampling = background_oversampling

        if selection is None:
            selection = self.available_selection

        selection = set(selection)

        if not selection.issubset(self.available_selection):
            difference = selection.difference(self.available_selection)
            raise ValueError(f"{difference} is not a valid method.")

        self.selection = selection

    @staticmethod
    def make_counts(geom, observation):
        """Make counts map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference map geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        counts : `~gammapy.maps.Map`
            Counts map.
        """
        counts = Map.from_geom(geom)
        counts.fill_events(observation.events)
        return counts

    @staticmethod
    def make_exposure(geom, observation):
        """Make exposure map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference map geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        """
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    @staticmethod
    def make_exposure_irf(geom, observation):
        """Make exposure map with irf geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        """
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    def make_background(self, geom, observation):
        """Make background map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        background : `~gammapy.maps.Map`
            Background map.
        """
        bkg_coordsys = observation.bkg.meta.get("FOVALIGN", "RADEC")

        if bkg_coordsys == "ALTAZ":
            pointing = observation.fixed_pointing_info
        elif bkg_coordsys == "RADEC":
            pointing = observation.pointing_radec
        else:
            raise ValueError(
                f"Invalid background coordinate system: {bkg_coordsys!r}\n"
                "Options: ALTAZ, RADEC"
            )

        return make_map_background_irf(
            pointing=pointing,
            ontime=observation.observation_time_duration,
            bkg=observation.bkg,
            geom=geom,
            oversampling=self.background_oversampling,
        )

    def make_edisp(self, geom, observation):
        """Make energy dispersion map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        edisp : `~gammapy.cube.EDispMap`
            Edisp map.
        """
        exposure = self.make_exposure_irf(geom.squash(axis="migra"), observation)

        return make_edisp_map(
            edisp=observation.edisp,
            pointing=observation.pointing_radec,
            geom=geom,
            exposure_map=exposure,
        )

    def make_edisp_kernel(self, geom, observation):
        """Make energy dispersion kernel map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom. Must contain "energy" and "energy_true" axes in that order.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        edisp : `~gammapy.cube.EDispKernelMap`
            EdispKernel map.
        """
        exposure = self.make_exposure_irf(geom.squash(axis="energy"), observation)

        return make_edisp_kernel_map(
            edisp=observation.edisp,
            pointing=observation.pointing_radec,
            geom=geom,
            exposure_map=exposure,
        )

    def make_psf(self, geom, observation):
        """Make psf map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        psf : `~gammapy.cube.PSFMap`
            Psf map.
        """
        psf = observation.psf
        if isinstance(psf, EnergyDependentMultiGaussPSF):
            rad_axis = geom.get_axis_by_name("theta")
            psf = psf.to_psf3d(rad=rad_axis.center)

        exposure = self.make_exposure_irf(geom.squash(axis="theta"), observation)

        return make_psf_map(
            psf=psf,
            pointing=observation.pointing_radec,
            geom=geom,
            exposure_map=exposure,
        )

    def make_meta_table(self, observation):
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
        meta_table["TELESCOP"] = [observation.aeff.meta["TELESCOP"]]
        meta_table["INSTRUME"] = [observation.aeff.meta["INSTRUME"]]
        meta_table["NAME"] = [observation.aeff.meta["CBD10001"][5:-1]]
        meta_table["OBS_ID"] = [observation.obs_id]
        #        NOT WORK YET
        #        info_table['AZ'] = [observation.pointing_altaz.az]
        #        info_table['ALT'] = [observation.pointing_altaz.alt]

        meta_table["RA_PNT"] = [observation.pointing_radec.icrs.ra.deg] * u.deg
        meta_table["DEC_PNT"] = [observation.pointing_radec.icrs.dec.deg] * u.deg

        return meta_table

    def run(self, dataset, observation):
        """Make map dataset.

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Reference dataset.
        observation : `~gammapy.data.Observation`
            Observation

        Returns
        -------
        dataset : `~gammapy.cube.MapDataset`
            Map dataset.
        """
        kwargs = {"gti": observation.gti}
        kwargs["meta_table"] = self.make_meta_table(observation)

        mask_safe = Map.from_geom(dataset.counts.geom, dtype=bool)
        mask_safe.data |= True

        kwargs["mask_safe"] = mask_safe

        if "counts" in self.selection:
            counts = self.make_counts(dataset.counts.geom, observation)
            kwargs["counts"] = counts

        if "exposure" in self.selection:
            exposure = self.make_exposure(dataset.exposure.geom, observation)
            kwargs["exposure"] = exposure

        if "background" in self.selection:
            background_map = self.make_background(dataset.counts.geom, observation)
            kwargs["models"] = BackgroundModel(
                background_map,
                name=dataset.name + "-bkg",
                datasets_names=[dataset.name],
            )

        if "psf" in self.selection:
            psf = self.make_psf(dataset.psf.psf_map.geom, observation)
            kwargs["psf"] = psf

        if "edisp" in self.selection:
            if dataset.edisp.edisp_map.geom.axes[0].name.upper() == "MIGRA":
                edisp = self.make_edisp(dataset.edisp.edisp_map.geom, observation)
            else:
                edisp = self.make_edisp_kernel(
                    dataset.edisp.edisp_map.geom, observation
                )

            kwargs["edisp"] = edisp

        return MapDataset(name=dataset.name, **kwargs)

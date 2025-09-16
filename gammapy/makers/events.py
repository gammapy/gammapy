# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import astropy.units as u
from astropy.table import Table
from scipy import integrate, interpolate
import numpy as np
from regions import PointSkyRegion
from gammapy.maps import MapAxis, ParallelLabelMapAxis
from gammapy.maps import RegionGeom, UnbinnedRegionGeom
from gammapy.makers import Maker, MapDatasetMaker
from gammapy.datasets import EventDataset
from gammapy.makers.utils import (
    make_edisp_kernel_map,
)


__all__ = ["EventDatasetMaker"]

log = logging.getLogger(__name__)


class EventDatasetMaker(Maker):
    """Make event dataset for a single IACT observation."""

    tag = "EventDatasetMaker"
    available_selection = ["exposure", "background", "psf", "edisp"]

    def __init__(
        self,
        selection=None,
        normalize_edisp_from_data_energy_space=True,  # THIS METHOD SEEMS MORE ACCURATE
        debug=False,
        **maker_kwargs,
    ):
        self.__debug = debug
        self._normalize_edisp_from_data_energy_space = (
            normalize_edisp_from_data_energy_space
        )
        if selection is None:
            selection = self.available_selection

        selection = set(selection)

        if not selection.issubset(self.available_selection):
            difference = selection.difference(self.available_selection)
            raise ValueError(f"{difference} is not a valid method.")

        self.selection = selection

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
        meta_table["TELESCOP"] = [observation.aeff.meta.get("TELESCOP", "Unknown")]
        meta_table["OBS_ID"] = [observation.obs_id]
        meta_table["RA_PNT"] = [observation.pointing.fixed_icrs.ra.deg] * u.deg
        meta_table["DEC_PNT"] = [observation.pointing.fixed_icrs.dec.deg] * u.deg

        return meta_table

    def run(self, dataset, observation):
        """Make the EventDataset.

        Parameters
        ----------
        dataset : EventDataset
            Empty EventDataset specifying the sky position from which computing the IRFs.
        observation : `~gammapy.data.Observation`
            Observation to build the EventDataset from

        Returns
        -------
        dataset : `~gammapy.datasets.EventDataset`
            EventDataset.
        """
        kwargs = {}
        kwargs["meta_table"] = self.make_meta_table(observation)
        if self.__debug:
            events = observation.events.select_time(
                [observation.gti.time_start, observation.gti.time_start + 0.3 * u.h]
            )
        else:
            events = observation.events
        kwargs["events"] = events

        parallel_labels = np.array(
            [
                events.table["EVENT_ID"].value,
                events.table["RA"].value,
                events.table["DEC"].value,
                events.table["ENERGY"].value,
            ]
        ).T.tolist()

        # energy_axis_true = MapAxis.from_energy_bounds(
        #    0.01, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true"
        # )
        energy_axis_true = observation.edisp.axes["energy_true"]

        geom = UnbinnedRegionGeom.create(
            region=PointSkyRegion(center=dataset.position),
            axes=[
                ParallelLabelMapAxis(
                    parallel_labels=parallel_labels,
                    parallel_names=["event_id", "ra", "dec", "energy"],
                    parallel_units=["", "deg", "deg", "TeV"],
                    name="events",
                ),
                energy_axis_true,
            ],
        )
        # geom.axes._n_spatial_axes = 0
        kwargs["geom"] = geom

        if "background" in self.selection:
            raise NotImplementedError("Background not implemented yet for unbinned")

        if "psf" in self.selection:
            raise NotImplementedError("PSF not implemented yet for unbinned")

        geom_irf_for_normalization = RegionGeom.create(
            region=PointSkyRegion(center=dataset.position),
            axes=[
                MapAxis.from_energy_bounds(
                    0.01, 50, nbin=20, per_decade=True, unit="TeV", name="energy"
                ),
                energy_axis_true,
            ],
        )
        kwargs["geom_normalization"] = geom_irf_for_normalization

        if "exposure" in self.selection:
            exposure = MapDatasetMaker.make_exposure(geom, observation)
            kwargs["exposure"] = exposure
            kwargs["exposure_original_irf"] = MapDatasetMaker.make_exposure(
                geom_irf_for_normalization.squash(axis_name="energy"), observation
            )

        if "edisp" in self.selection:
            edisp = self.make_edisp_kernel(observation, geom=geom)
            edisp_original_irf = self.make_edisp_kernel(
                observation, geom=geom_irf_for_normalization, original_irf=True
            )

            # THIS SECOND PART IS BETTER AT LOW ENERGY AND IS USED IN THE NOTEBOOK
            axis = edisp.edisp_map.geom.axes.index_data("energy")
            if self._normalize_edisp_from_data_energy_space:
                args_sorted = np.argsort(geom.axes["events"]["energy"].center)
                energy_reco_sorted = geom.axes["events"]["energy"].center[args_sorted]
                edisp_sorted = np.take(edisp.edisp_map.data, args_sorted, axis=axis)
                normalization = integrate.simpson(
                    edisp_sorted, energy_reco_sorted, axis=axis
                )
            else:
                interpolation = interpolate.interp1d(
                    geom.axes["events"]["energy"].center,
                    edisp.edisp_map.quantity,
                    axis=axis,
                    kind="linear",
                    fill_value="extrapolate",
                )
                normalization = integrate.simpson(
                    interpolation(geom_irf_for_normalization.axes["energy"].center),
                    geom_irf_for_normalization.axes["energy"].center,
                    axis=axis,
                )
            edisp.edisp_map.quantity = (
                np.nan_to_num(
                    np.einsum("trxy,txy->trxy", edisp.edisp_map.data, normalization**-1)
                )
                * edisp.edisp_map.unit
            )
            kwargs["edisp"] = edisp
            kwargs["edisp_original_irf"] = edisp_original_irf

        # dataset = self.map_ds_maker.run(emptyMapDs, obs)
        #
        # if self.safe_mask_maker:
        #    dataset = self.safe_mask_maker.run(dataset, obs)
        #    kwargs["mask_safe"] = dataset.mask_safe
        #
        # for key in self.selection:
        #    kwargs[key] = getattr(dataset, key, None)

        kwargs["gti"] = dataset.gti

        return EventDataset(name=dataset.name, **kwargs)

    # def make_psf(self, observation):
    #    """Make PSF map.
    #
    #    Parameters
    #    ----------
    #    geom : `~gammapy.maps.Geom`
    #        Reference geometry.
    #    observation : `~gammapy.data.Observation`
    #        Observation container.
    #
    #    Returns
    #    -------
    #    psf : `~gammapy.irf.PSFMap`
    #        PSF map.
    #    """
    #    psf = observation.psf
    #
    #    geom = psf.psf_map.geom
    #
    #    if isinstance(psf, RecoPSFMap):
    #        return RecoPSFMap(psf.psf_map.interp_to_geom(geom))
    #    elif isinstance(psf, PSFMap):
    #        return PSFMap(psf.psf_map.interp_to_geom(geom))
    #    exposure = self.make_exposure_irf(geom.squash(axis_name="rad"), observation)
    #
    #    return make_psf_map(
    #        psf=psf,
    #        pointing=observation.get_pointing_icrs(observation.tmid),
    #        geom=geom,
    #        exposure_map=exposure,
    #    )

    #
    # def make_edisp(self, observation, geom):
    #    """Make energy dispersion per events.
    #
    #    Parameters
    #    ----------
    #    observation : `~gammapy.data.Observation`
    #        Observation container.
    #
    #    Returns
    #    -------
    #    edisp : `~gammapy.irf.EDispKernelMap`
    #        Energy dispersion map.
    #    """
    #    exposure = MapDatasetMaker.make_exposure_irf(geom, observation) #exposure for true_energy axis -> binned
    #    #use_region_center = getattr(self, "use_region_center", True)
    #    return make_edisp_map(
    #        edisp=observation.edisp,
    #        pointing=observation.get_pointing_icrs(observation.tmid),
    #        geom=geom,
    #        exposure_map=exposure,
    #        use_region_center=True,
    #    )
    #

    def make_edisp_kernel(self, observation, geom, original_irf=False):
        if original_irf:
            exposure = MapDatasetMaker.make_exposure_irf(
                geom.squash(axis_name="energy"), observation
            )
        else:
            exposure = MapDatasetMaker.make_exposure_irf(
                geom, observation
            )  # Squash over events?
        use_region_center = getattr(self, "use_region_center", True)
        return make_edisp_kernel_map(
            edisp=observation.edisp,
            pointing=observation.get_pointing_icrs(observation.tmid),
            geom=geom,
            exposure_map=exposure,
            use_region_center=use_region_center,
        )

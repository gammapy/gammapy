# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import astropy.units as u
from astropy.table import Table
from regions import PointSkyRegion
from gammapy.datasets import MapDatasetMetaData
from gammapy.irf import EDispKernelMap, PSFMap, RecoPSFMap
from gammapy.maps import Map
from .core import Maker
from .utils import (
    make_counts_rad_max,
    make_edisp_kernel_map,
    make_edisp_map,
    make_map_background_irf,
    make_map_exposure_true_energy,
    make_psf_map,
)

__all__ = ["MapDatasetMaker"]

log = logging.getLogger(__name__)


class MapDatasetMaker(Maker):
    """Make binned maps for a single IACT observation.

    Parameters
    ----------
    selection : list of str, optional
        Select which maps to make, the available options are:
        'counts', 'exposure', 'background', 'psf', 'edisp'.
        By default, all maps are made.
    background_oversampling : int
        Background evaluation oversampling factor in energy.
    background_interp_missing_data : bool, optional
        Interpolate missing values in background 3d map.
        Default is True, have to be set to True for CTA IRF.
    background_pad_offset : bool, optional
        Pad one bin in offset for 2d background map.
        This avoids extrapolation at edges and use the nearest value.
        Default is True.

    Examples
    --------
    This example shows how to run the MapMaker for a single observation.

    >>> from gammapy.data import DataStore
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.maps import WcsGeom, MapAxis
    >>> from gammapy.makers import MapDatasetMaker

    >>> # Load an observation
    >>> data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    >>> obs = data_store.obs(23523)

    >>> # Prepare the geometry
    >>> energy_axis = MapAxis.from_energy_bounds(1.0, 10.0, 4, unit="TeV")
    >>> energy_axis_true = MapAxis.from_energy_bounds( 0.5, 20, 10, unit="TeV", name="energy_true")
    >>> geom = WcsGeom.create(
    ...        skydir=(83.633, 22.014),
    ...        binsz=0.02,
    ...        width=(2, 2),
    ...        frame="icrs",
    ...        proj="CAR",
    ...        axes=[energy_axis],
    ...    )

    >>> # Run the maker
    >>> empty = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true, name="empty")
    >>> maker = MapDatasetMaker()
    >>> dataset = maker.run(empty, obs)
    >>> print(dataset)
    MapDataset
    ----------
    <BLANKLINE>
      Name                            : empty
    <BLANKLINE>
      Total counts                    : 787
      Total background counts         : 684.52
      Total excess counts             : 102.48
    <BLANKLINE>
      Predicted counts                : 684.52
      Predicted background counts     : 684.52
      Predicted excess counts         : nan
    <BLANKLINE>
      Exposure min                    : 7.01e+07 m2 s
      Exposure max                    : 1.10e+09 m2 s
    <BLANKLINE>
      Number of total bins            : 40000
      Number of fit bins              : 40000
    <BLANKLINE>
      Fit statistic type              : cash
      Fit statistic value (-2 log(L)) : nan
    <BLANKLINE>
      Number of models                : 0
      Number of parameters            : 0
      Number of free parameters       : 0

    """

    tag = "MapDatasetMaker"
    available_selection = ["counts", "exposure", "background", "psf", "edisp"]

    def __init__(
        self,
        selection=None,
        background_oversampling=None,
        background_interp_missing_data=True,
        background_pad_offset=True,
    ):
        self.background_oversampling = background_oversampling
        self.background_interp_missing_data = background_interp_missing_data
        self.background_pad_offset = background_pad_offset
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
            Reference map geometry.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        counts : `~gammapy.maps.Map`
            Counts map.
        """
        if geom.is_region and isinstance(geom.region, PointSkyRegion):
            counts = make_counts_rad_max(geom, observation.rad_max, observation.events)
        else:
            counts = Map.from_geom(geom)
            counts.fill_events(observation.events)
        return counts

    @staticmethod
    def make_exposure(geom, observation, use_region_center=True):
        """Make exposure map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference map geometry.
        observation : `~gammapy.data.Observation`
            Observation container.
        use_region_center : bool, optional
            For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
            If False, average over the whole region.
            Default is True.

        Returns
        -------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        """
        if isinstance(observation.aeff, Map):
            return observation.aeff.interp_to_geom(
                geom=geom,
            )
        return make_map_exposure_true_energy(
            pointing=observation.get_pointing_icrs(observation.tmid),
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
            use_region_center=use_region_center,
        )

    @staticmethod
    def make_exposure_irf(geom, observation, use_region_center=True):
        """Make exposure map with IRF geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geometry.
        observation : `~gammapy.data.Observation`
            Observation container.
        use_region_center : bool, optional
            For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
            If False, average over the whole region.
            Default is True.

        Returns
        -------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        """
        return make_map_exposure_true_energy(
            pointing=observation.get_pointing_icrs(observation.tmid),
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
            use_region_center=use_region_center,
        )

    def make_background(self, geom, observation):
        """Make background map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geometry.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        background : `~gammapy.maps.Map`
            Background map.
        """
        bkg = observation.bkg

        if isinstance(bkg, Map):
            return bkg.interp_to_geom(geom=geom, preserve_counts=True)

        use_region_center = getattr(self, "use_region_center", True)

        if self.background_interp_missing_data:
            bkg.interp_missing_data(axis_name="energy")

        if self.background_pad_offset and bkg.has_offset_axis:
            bkg = bkg.pad(1, mode="edge", axis_name="offset")

        return make_map_background_irf(
            pointing=observation.pointing,
            ontime=observation.observation_time_duration,
            bkg=bkg,
            geom=geom,
            oversampling=self.background_oversampling,
            use_region_center=use_region_center,
            obstime=observation.tmid,
        )

    def make_edisp(self, geom, observation):
        """Make energy dispersion map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geometry.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        edisp : `~gammapy.irf.EDispMap`
            Energy dispersion map.
        """
        exposure = self.make_exposure_irf(geom.squash(axis_name="migra"), observation)

        use_region_center = getattr(self, "use_region_center", True)

        return make_edisp_map(
            edisp=observation.edisp,
            pointing=observation.get_pointing_icrs(observation.tmid),
            geom=geom,
            exposure_map=exposure,
            use_region_center=use_region_center,
        )

    def make_edisp_kernel(self, geom, observation):
        """Make energy dispersion kernel map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geometry. Must contain "energy" and "energy_true" axes in that order.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        edisp : `~gammapy.irf.EDispKernelMap`
            Energy dispersion kernel map.
        """
        if isinstance(observation.edisp, EDispKernelMap):
            exposure = None
            interp_map = observation.edisp.edisp_map.interp_to_geom(geom)
            return EDispKernelMap(edisp_kernel_map=interp_map, exposure_map=exposure)

        exposure = self.make_exposure_irf(geom.squash(axis_name="energy"), observation)

        use_region_center = getattr(self, "use_region_center", True)

        return make_edisp_kernel_map(
            edisp=observation.edisp,
            pointing=observation.get_pointing_icrs(observation.tmid),
            geom=geom,
            exposure_map=exposure,
            use_region_center=use_region_center,
        )

    def make_psf(self, geom, observation):
        """Make PSF map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geometry.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        psf : `~gammapy.irf.PSFMap`
            PSF map.
        """
        psf = observation.psf

        if isinstance(psf, RecoPSFMap):
            return RecoPSFMap(psf.psf_map.interp_to_geom(geom))
        elif isinstance(psf, PSFMap):
            return PSFMap(psf.psf_map.interp_to_geom(geom))
        exposure = self.make_exposure_irf(geom.squash(axis_name="rad"), observation)

        return make_psf_map(
            psf=psf,
            pointing=observation.get_pointing_icrs(observation.tmid),
            geom=geom,
            exposure_map=exposure,
        )

    @staticmethod
    def make_meta_table(observation):
        """Make information meta table.

        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation.

        Returns
        -------
        meta_table : `~astropy.table.Table`
            Meta table.
        """
        row = {}
        row["TELESCOP"] = observation.aeff.meta.get("TELESCOP", "Unknown")
        row["OBS_ID"] = observation.obs_id

        row.update(observation.pointing.to_fits_header())

        meta_table = Table([row])
        if "ALT_PNT" in meta_table.colnames:
            meta_table["ALT_PNT"].unit = u.deg
            meta_table["AZ_PNT"].unit = u.deg
        if "RA_PNT" in meta_table.colnames:
            meta_table["RA_PNT"].unit = u.deg
            meta_table["DEC_PNT"].unit = u.deg

        return meta_table

    @staticmethod
    def _make_metadata(table):
        return MapDatasetMetaData._from_meta_table(table)

    def run(self, dataset, observation):
        """Make map dataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Reference dataset.
        observation : `~gammapy.data.Observation`
            Observation.

        Returns
        -------
        dataset : `~gammapy.datasets.MapDataset`
            Map dataset.
        """
        kwargs = {"gti": observation.gti}
        kwargs["meta_table"] = self.make_meta_table(observation)
        kwargs["meta"] = self._make_metadata(kwargs["meta_table"])

        mask_safe = Map.from_geom(dataset.counts.geom, dtype=bool)
        mask_safe.data[...] = True

        kwargs["mask_safe"] = mask_safe

        if "counts" in self.selection:
            counts = self.make_counts(dataset.counts.geom, observation)
        else:
            counts = Map.from_geom(dataset.counts.geom, data=0)
        kwargs["counts"] = counts

        if "exposure" in self.selection:
            exposure = self.make_exposure(dataset.exposure.geom, observation)
            kwargs["exposure"] = exposure

        if "background" in self.selection:
            kwargs["background"] = self.make_background(
                dataset.counts.geom, observation
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

        return dataset.__class__(name=dataset.name, **kwargs)

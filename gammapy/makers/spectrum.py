# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from regions import CircleSkyRegion
from .map import MapDatasetMaker

__all__ = ["SpectrumDatasetMaker"]

log = logging.getLogger(__name__)


class SpectrumDatasetMaker(MapDatasetMaker):
    """Make spectrum for a single IACT observation.

    The IRFs and background are computed at a single fixed offset,
    which is recommended only for point-sources.

    Parameters
    ----------
    selection : list of str, optional
        Select which maps to make, the available options are:
        'counts', 'exposure', 'background', 'edisp'.
        By default, all maps are made.
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    background_oversampling : int
        Background evaluation oversampling factor in energy.
    use_region_center : bool
        If True, approximate the IRFs by the value at the center of the region.
        If False, the IRFs are averaged over the entire.
    """

    tag = "SpectrumDatasetMaker"
    available_selection = ["counts", "background", "exposure", "edisp"]

    def __init__(
        self,
        selection=None,
        containment_correction=False,
        background_oversampling=None,
        use_region_center=True,
    ):
        self.containment_correction = containment_correction
        self.use_region_center = use_region_center
        super().__init__(
            selection=selection, background_oversampling=background_oversampling
        )

    def make_exposure(self, geom, observation):
        """Make exposure.

        Parameters
        ----------
        geom : `~gammapy.maps.RegionGeom`
            Reference map geometry.
        observation : `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        exposure : `~gammapy.maps.RegionNDMap`
            Exposure map.
        """
        exposure = super().make_exposure(
            geom, observation, use_region_center=self.use_region_center
        )

        is_pointlike = exposure.meta.get("is_pointlike", False)
        if is_pointlike and self.use_region_center is False:
            log.warning(
                "MapMaker: use_region_center=False should not be used with point-like IRF. "
                "Results are likely inaccurate."
            )

        if self.containment_correction:
            if is_pointlike:
                raise ValueError(
                    "Cannot apply containment correction for point-like IRF."
                )

            if not isinstance(geom.region, CircleSkyRegion):
                raise TypeError(
                    "Containment correction only supported for circular regions."
                )
            offset = geom.separation(observation.get_pointing_icrs(observation.tmid))
            containment = observation.psf.containment(
                rad=geom.region.radius,
                offset=offset,
                energy_true=geom.axes["energy_true"].center,
            )
            exposure.quantity *= containment.reshape(geom.data_shape)

        return exposure

    @staticmethod
    def make_counts(geom, observation):
        """Make counts map.

        If the `~gammapy.maps.RegionGeom` is built from a `~regions.CircleSkyRegion`,
        the latter will be directly used to extract the counts.
        If instead the `~gammapy.maps.RegionGeom` is built from a `~regions.PointSkyRegion`,
        the size of the ON region is taken from the `RAD_MAX_2D` table containing energy-dependent theta2 cuts.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference map geometry.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        counts : `~gammapy.maps.RegionNDMap`
            Counts map.
        """
        return super(SpectrumDatasetMaker, SpectrumDatasetMaker).make_counts(
            geom, observation
        )

    def run(self, dataset, observation):
        """Make spectrum dataset.

        Parameters
        ----------
        dataset : `~gammapy.spectrum.SpectrumDataset`
            Reference dataset.
        observation : `~gammapy.data.Observation`
            Observation.

        Returns
        -------
        dataset : `~gammapy.spectrum.SpectrumDataset`
            Spectrum dataset.
        """
        return super(SpectrumDatasetMaker, self).run(dataset, observation)

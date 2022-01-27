# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from abc import ABCMeta, abstractmethod

from astropy import units as u
from astropy.coordinates import Angle
from regions import PixCoord, PointSkyRegion
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.maps import RegionGeom, RegionNDMap, WcsGeom, WcsNDMap
from ..core import Maker
from ..utils import make_counts_off_rad_max

__all__ = ["ReflectedRegionsFinder", "ReflectedRegionsBackgroundMaker"]

log = logging.getLogger(__name__)

FULL_CIRCLE = Angle(2 * np.pi, "rad")


class RegionsFinder(metaclass=ABCMeta):
    '''Baseclass for regions finders'''

    @abstractmethod
    def run(self, region, center, exclusion_mask=None):
        """Find regions to calculate background counts.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to rotate
        center : `~astropy.coordinates.SkyCoord`
            Rotation point
        exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
            Exclusion mask

        Returns
        -------
        regions : list of `SkyRegion`
            Reflected regions
        wcs: `~astropy.wcs.WCS`
            WCS for the determined regions
        """


class ReflectedRegionsFinder(RegionsFinder):
    """Find reflected regions.

    This class is responsible for placing a reflected region for a given
    input region and pointing position. It converts to pixel coordinates
    internally assuming a tangent projection at center position.

    If the center lies inside the input region, no reflected regions
    can be found.

    If you want to make a
    background estimate for an IACT observation using the reflected regions
    method, see also `~gammapy.makers.ReflectedRegionsBackgroundMaker`

    Parameters
    ----------
    angle_increment : `~astropy.coordinates.Angle`, optional
        Rotation angle applied when a region falls in an excluded region.
    min_distance : `~astropy.coordinates.Angle`, optional
        Minimal distance between two consecutive reflected regions
    min_distance_input : `~astropy.coordinates.Angle`, optional
        Minimal distance from input region
    max_region_number : int, optional
        Maximum number of regions to use
    binsz : `~astropy.coordinates.Angle`
        Bin size of the reference map used for region finding.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord, Angle
    >>> from regions import CircleSkyRegion
    >>> from gammapy.makers import ReflectedRegionsFinder
    >>> pointing = SkyCoord(83.2, 22.7, unit='deg', frame='icrs')
    >>> target_position = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
    >>> theta = Angle(0.4, 'deg')
    >>> on_region = CircleSkyRegion(target_position, theta)
    >>> finder = ReflectedRegionsFinder(min_distance_input='1 rad')
    >>> regions, wcs = finder.run(region=on_region, center=pointing)
    >>> print(regions[0]) # doctest: +SKIP
    Region: CircleSkyRegion
    center: <SkyCoord (ICRS): (ra, dec) in deg
        (83.19879005, 25.57300957)>
    radius: 1438.320341895 arcsec
    """

    def __init__(
        self,
        angle_increment="0.1 rad",
        min_distance="0 rad",
        min_distance_input="0.1 rad",
        max_region_number=10000,
        binsz="0.01 deg",
    ):

        self.angle_increment = Angle(angle_increment)

        if self.angle_increment <= Angle(0, "deg"):
            raise ValueError("angle_increment is too small")

        self.min_distance = Angle(min_distance)
        self.min_distance_input = Angle(min_distance_input)

        self.max_region_number = max_region_number
        self.binsz = Angle(binsz)

    def _create_reference_geometry(self, region, center):
        """Reference geometry

        The size of the map is chosen such that all reflected regions are
        contained on the image.
        To do so, the reference map width is taken to be 4 times the distance between
        the target region center and the rotation point. This distance is larger than
        the typical dimension of the region itself (otherwise the rotation point would
        lie inside the region). A minimal width value is added by default in case the
        region center and the rotation center are too close.

        The WCS of the map is the TAN projection at the `center` in the coordinate
        system used by the `region` center.
        """
        frame = region.center.frame.name

        # width is the full width of an image (not the radius)
        width = 4 * region.center.separation(center) + Angle("0.3 deg")

        return WcsGeom.create(
            skydir=center, binsz=self.binsz, width=width, frame=frame, proj="TAN"
        )

    @staticmethod
    def _get_center_pixel(center, reference_geom):
        """Center pix coordinate"""
        return PixCoord.from_sky(center, reference_geom.wcs)

    @staticmethod
    def _get_region_pixels(region, reference_geom):
        """Pixel region"""
        return region.to_pixel(reference_geom.wcs)

    @staticmethod
    def _region_angular_size(region, reference_geom, center_pix):
        """Compute maximum angular size of a group of pixels as seen from center.

        This assumes that the center lies outside the group of pixel

        Returns
        -------
        angular_size : `~astropy.coordinates.Angle`
            the maximum angular size
        """
        mask = reference_geom.region_mask([region]).data
        pix_y, pix_x = np.nonzero(mask)

        pixels = PixCoord(pix_x, pix_y)

        dx, dy = center_pix.x - pixels.x, center_pix.y - pixels.y
        angles = Angle(np.arctan2(dx, dy), "rad")
        angular_size = np.max(angles) - np.min(angles)

        if angular_size.value > np.pi:
            angle_wrapped = angles.wrap_at(0 * u.rad)
            angular_size = np.max(angle_wrapped) - np.min(angle_wrapped)

        return angular_size

    @staticmethod
    def _exclusion_mask_ref(reference_geom, exclusion_mask):
        """Exclusion mask reprojected"""
        if exclusion_mask:
            mask = exclusion_mask.interp_to_geom(reference_geom, fill_value=True)
        else:
            mask = WcsNDMap.from_geom(geom=reference_geom, data=True)
        return mask

    @staticmethod
    def _get_excluded_pixels(reference_geom, exclusion_mask):
        """Excluded pix coords"""
        # find excluded PixCoords
        exclusion_mask = ReflectedRegionsFinder._exclusion_mask_ref(
            reference_geom, exclusion_mask,
        )
        pix_y, pix_x = np.nonzero(~exclusion_mask.data)
        return PixCoord(pix_x, pix_y)

    def _get_angle_range(self, region, reference_geom, center_pix):
        """Minimum and maximum angle"""
        region_angular_size = self._region_angular_size(
            region=region, reference_geom=reference_geom, center_pix=center_pix
        )
        # Minimum angle a region has to be moved to not overlap with previous one
        # Add required minimal distance between two off regions
        angle_min = region_angular_size + self.min_distance
        angle_max =  FULL_CIRCLE - angle_min - self.min_distance_input
        return angle_min, angle_max

    def run(self, region, center, exclusion_mask=None):
        """Find reflected regions.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to rotate
        center : `~astropy.coordinates.SkyCoord`
            Rotation point

        Returns
        -------
        regions : list of `SkyRegion`
            Reflected regions
        wcs: `~astropy.wcs.WCS`
            WCS for the determined regions
        """
        regions = []

        reference_geom = self._create_reference_geometry(region, center)
        center_pixel = self._get_center_pixel(center, reference_geom)
        angle_min, angle_max = self._get_angle_range(
            region=region, reference_geom=reference_geom, center_pix=center_pixel,
        )

        region_pix = self._get_region_pixels(region, reference_geom)
        excluded_pixels = self._get_excluded_pixels(reference_geom, exclusion_mask)

        angle = angle_min + self.min_distance_input
        while angle < angle_max:
            region_test = region_pix.rotate(center_pixel, angle)

            if not np.any(region_test.contains(excluded_pixels)):
                region = region_test.to_sky(reference_geom.wcs)
                regions.append(region)

                if len(regions) >= self.max_region_number:
                    break

                angle += angle_min
            else:
                angle += self.angle_increment

        return regions, reference_geom.wcs


class ReflectedRegionsBackgroundMaker(Maker):
    """Reflected regions background maker.

    Attributes
    ----------
    regions_finder: RegionsFinder
        if not given, a `ReflectedRegionsFinder` will be created and
        any of the ``**kwargs`` will be forwarded to the `ReflectedRegionsFinder`.
    exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
        Exclusion mask
    """

    tag = "ReflectedRegionsBackgroundMaker"

    def __init__(
        self,
        region_finder=None,
        exclusion_mask=None,
        **kwargs,
    ):

        if exclusion_mask and not exclusion_mask.is_mask:
            raise ValueError("Exclusion mask must contain boolean values")

        self.exclusion_mask = exclusion_mask

        if region_finder is None:
            self.region_finder = ReflectedRegionsFinder(**kwargs)
        else:
            if len(kwargs) != 0:
                raise ValueError('No kwargs can be given if providing a region_finder')
            self.region_finder = region_finder

    def make_counts_off(self, dataset, observation):
        """Make off counts.

        **NOTE for 1D analysis:** as for
        `~gammapy.makers.map.MapDatasetMaker.make_counts`,
        if the geometry of the dataset is a `~regions.CircleSkyRegion` then only
        a single instance of the `ReflectedRegionsFinder` will be called.
        If, on the other hand, the geometry of the dataset is a
        `~regions.PointSkyRegion`, then we have to call the
        `ReflectedRegionsFinder` several time, each time with a different size
        of the on region that we will read from the `RAD_MAX_2D` table.

        Parameters
        ----------
        dataset : `~gammapy.datasets.SpectrumDataset`
            Spectrum dataset.
        observation : `~gammapy.observation.Observation`
            Observation container.

        Returns
        -------
        counts_off : `~gammapy.maps.RegionNDMap`
            Counts vs estimated energy extracted from the OFF regions.
        """
        if dataset.counts.geom.is_region and isinstance(
            dataset.counts.geom.region, PointSkyRegion
        ):
            counts_off, acceptance_off = make_counts_off_rad_max(
                geom=dataset.counts.geom,
                rad_max=observation.rad_max,
                events=observation.events,
                region_finder=self.region_finder,
                exclusion_mask=self.exclusion_mask,
            )

        else:
            regions, wcs = self.region_finder.run(
                center=observation.pointing_radec,
                region=dataset.counts.geom.region,
                exclusion_mask=self.exclusion_mask,
            )

            energy_axis = dataset.counts.geom.axes["energy"]

            if len(regions) > 0:
                geom = RegionGeom.from_regions(
                    regions=regions,
                    axes=[energy_axis],
                    wcs=wcs,
                )

                counts_off = RegionNDMap.from_geom(geom=geom)
                counts_off.fill_events(observation.events)
                acceptance_off = RegionNDMap.from_geom(geom=geom, data=len(regions))
            else:
                # if no OFF regions are found, off is set to None and acceptance_off to zero
                log.warning(
                    f"ReflectedRegionsBackgroundMaker failed. No OFF region found outside exclusion mask for {dataset.name}."
                )

                counts_off = None
                acceptance_off = RegionNDMap.from_geom(geom=dataset._geom, data=0)

        return counts_off, acceptance_off

    def run(self, dataset, observation):
        """Run reflected regions background maker

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset.
        observation : `DatastoreObservation`
            Data store observation.

        Returns
        -------
        dataset_on_off : `SpectrumDatasetOnOff`
            On off dataset.
        """
        counts_off, acceptance_off = self.make_counts_off(dataset, observation)
        acceptance = RegionNDMap.from_geom(geom=dataset.counts.geom, data=1)

        dataset_onoff = SpectrumDatasetOnOff.from_spectrum_dataset(
            dataset=dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            counts_off=counts_off,
            name=dataset.name,
        )

        if dataset_onoff.counts_off is None:
            dataset_onoff.mask_safe.data[...] = False
            log.warning(
                f"ReflectedRegionsBackgroundMaker failed. Setting {dataset_onoff.name} mask to False."
            )
        return dataset_onoff

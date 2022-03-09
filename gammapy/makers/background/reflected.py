# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from itertools import combinations
from abc import ABCMeta, abstractmethod

from astropy import units as u
from astropy.coordinates import Angle
from regions import PixCoord, PointSkyRegion
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.maps import RegionGeom, RegionNDMap, WcsGeom, WcsNDMap
from ..core import Maker
from ..utils import make_counts_off_rad_max

__all__ = [
    "ReflectedRegionsBackgroundMaker",
    "ReflectedRegionsFinder",
    "RegionsFinder",
    "WobbleRegionsFinder",
]

log = logging.getLogger(__name__)

FULL_CIRCLE = Angle(2 * np.pi, "rad")


class RegionsFinder(metaclass=ABCMeta):
    '''Baseclass for regions finders


    Parameters
    ----------
    binsz : `~astropy.coordinates.Angle`
        Bin size of the reference map used for region finding.
    '''
    def __init__(self, binsz=0.01 * u.deg):
        '''Create a new RegionFinder'''
        self.binsz = Angle(binsz)

    @abstractmethod
    def run(self, region, center, exclusion_mask=None):
        """Find reflected regions.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to rotate
        center : `~astropy.coordinates.SkyCoord`
            Rotation point
        exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
            Exclusion mask. Regions intersecting with this mask will not be
            included in the returned regions.

        Returns
        -------
        regions : list of `~regions.SkyRegion`
            Reflected regions
        wcs: `~astropy.wcs.WCS`
            WCS for the determined regions
        """

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



class WobbleRegionsFinder(RegionsFinder):
    """Find the OFF regions symmetric to the ON region

    This is a simpler version of the `ReflectedRegionsFinder`, that
    will place ``n_off_regions`` regions at symmetric positions on the
    circle created by the center position and the on region.

    Returns no regions if the regions are found to be overlapping,
    in that case reduce the number of off regions and/or their size.

    Parameters
    ----------
    n_off_regions: int
        Number of off regions to create. Actual number of off regions
        might be smaller if an ``exclusion_mask`` is given to `WobbleRegionsFinder.run`
    binsz : `~astropy.coordinates.Angle`
        Bin size of the reference map used for region finding.
    """
    def __init__(self, n_off_regions, binsz=0.01 * u.deg):
        super().__init__(binsz=binsz)
        self.n_off_regions = n_off_regions

    def run(self, region, center, exclusion_mask=None):
        """Find off regions.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to rotate
        center : `~astropy.coordinates.SkyCoord`
            Rotation point
        exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
            Exclusion mask. Regions intersecting with this mask will not be
            included in the returned regions.

        Returns
        -------
        regions : list of `~regions.SkyRegion`
            Reflected regions
        wcs: `~astropy.wcs.WCS`
            WCS for the determined regions
        """
        reference_geom = self._create_reference_geometry(region, center)
        center_pixel = self._get_center_pixel(center, reference_geom)

        region_pix = self._get_region_pixels(region, reference_geom)
        excluded_pixels = self._get_excluded_pixels(reference_geom, exclusion_mask)

        n_positions = self.n_off_regions + 1
        increment = FULL_CIRCLE / n_positions

        regions = []
        for i in range(1, n_positions):
            angle = i * increment
            region_test = region_pix.rotate(center_pixel, angle)

            # for PointSkyRegion, we test if the point is inside the exclusion mask
            # otherwise we test if there is overlap

            excluded = False
            if exclusion_mask is not None:
                if isinstance(region, PointSkyRegion):
                    excluded = (excluded_pixels.separation(region_test.center) < 1).any()
                else:
                    excluded = region_test.contains(excluded_pixels).any()

            if not excluded:
                regions.append(region_test)


        # We cannot check for overlap of PointSkyRegion here, this is done later
        # in make_counts_off_rad_max in the rad_max case
        if not isinstance(region, PointSkyRegion):
            if self._are_regions_overlapping(regions, reference_geom):
                log.warning('Found overlapping off regions, returning no regions')
                return [], reference_geom.wcs

        return [r.to_sky(reference_geom.wcs) for r in regions], reference_geom.wcs


    @staticmethod
    def _are_regions_overlapping(regions, reference_geom):
        # check for overl
        masks = [
            region.to_mask().to_image(reference_geom._shape) > 0
            for region in regions
        ]
        for mask_a, mask_b in combinations(masks, 2):
            if np.any(mask_a & mask_b):
                return True

        return False


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
    >>> print(regions[0]) # doctest: +ELLIPSIS
    Region: CircleSkyRegion
    center: <SkyCoord (ICRS): (ra, dec) in deg
        (83.19879005, 25.57300957)>
    radius: 1438.3... arcsec
    """

    def __init__(
        self,
        angle_increment="0.1 rad",
        min_distance="0 rad",
        min_distance_input="0.1 rad",
        max_region_number=10000,
        binsz="0.01 deg",
    ):
        super().__init__(binsz=binsz)
        self.angle_increment = Angle(angle_increment)

        if self.angle_increment <= Angle(0, "deg"):
            raise ValueError("angle_increment is too small")

        self.min_distance = Angle(min_distance)
        self.min_distance_input = Angle(min_distance_input)

        self.max_region_number = max_region_number
        self.binsz = Angle(binsz)

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
        exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
            Exclusion mask. Regions intersecting with this mask will not be
            included in the returned regions.

        Returns
        -------
        regions : list of `~regions.SkyRegion`
            Reflected regions
        wcs: `~astropy.wcs.WCS`
            WCS for the determined regions
        """
        if isinstance(region, PointSkyRegion):
            raise TypeError(
                '`ReflectedRegionsFinder` does not work for `PointSkyRegion`'
                ', use `WobbleRegionsFinder` instead'
            )

        regions = []

        reference_geom = self._create_reference_geometry(region, center)
        center_pixel = self._get_center_pixel(center, reference_geom)

        region_pix = self._get_region_pixels(region, reference_geom)
        excluded_pixels = self._get_excluded_pixels(reference_geom, exclusion_mask)

        angle_min, angle_max = self._get_angle_range(
            region=region, reference_geom=reference_geom, center_pix=center_pixel,
        )

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
        on_geom = dataset.counts.geom
        if observation.rad_max is not None:
            if not isinstance(on_geom.region, PointSkyRegion):
                raise ValueError('Must use PointSkyRegion on region in point-like analysis')

            counts_off, acceptance_off = make_counts_off_rad_max(
                on_geom=on_geom,
                rad_max=observation.rad_max,
                events=observation.events,
                region_finder=self.region_finder,
                exclusion_mask=self.exclusion_mask,
            )

        else:
            regions, wcs = self.region_finder.run(
                center=observation.pointing_radec,
                region=on_geom.region,
                exclusion_mask=self.exclusion_mask,
            )

            energy_axis = on_geom.axes["energy"]

            if len(regions) > 0:
                off_geom = RegionGeom.from_regions(
                    regions=regions,
                    axes=[energy_axis],
                    wcs=wcs,
                )

                counts_off = RegionNDMap.from_geom(geom=off_geom)
                counts_off.fill_events(observation.events)
                acceptance_off = RegionNDMap.from_geom(geom=off_geom, data=len(regions))
            else:
                # if no OFF regions are found, off is set to None and acceptance_off to zero
                log.warning(
                    f"ReflectedRegionsBackgroundMaker failed. No OFF region found outside exclusion mask for {dataset.name}."
                )

                counts_off = None
                acceptance_off = RegionNDMap.from_geom(geom=on_geom, data=0)

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

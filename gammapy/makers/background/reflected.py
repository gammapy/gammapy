# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from abc import ABCMeta, abstractmethod
from itertools import combinations
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from regions import CircleSkyRegion, PixCoord, PointSkyRegion
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


def are_regions_overlapping_rad_max(regions, rad_max, offset, e_min, e_max):
    """
    Calculate pair-wise separations between all regions and compare with rad_max
    to find overlaps.
    """
    separations = u.Quantity(
        [a.center.separation(b.center) for a, b in combinations(regions, 2)]
    )

    rad_max_at_offset = rad_max.evaluate(offset=offset)
    # do not check bins outside of energy range
    edges_min = rad_max.axes["energy"].edges_min
    edges_max = rad_max.axes["energy"].edges_max
    # to be sure all possible values are included, we check
    # for the *upper* energy bin to be larger than e_min and the *lower* edge
    # to be larger than e_max
    mask = (edges_max >= e_min) & (edges_min <= e_max)
    rad_max_at_offset = rad_max_at_offset[mask]
    return np.any(separations[np.newaxis, :] < (2 * rad_max_at_offset))


def is_rad_max_compatible_region_geom(rad_max, geom, rtol=1e-3):
    """Check if input RegionGeom is compatible with rad_max for point-like analysis.

    Parameters
    ----------
    geom : `~gammapy.maps.RegionGeom`
        input RegionGeom.
    rtol : float
        relative tolerance

    Returns
    -------
    valid : bool
        True if rad_max is fixed and region is a CircleSkyRegion with compatible radius
        True if region is a PointSkyRegion
        False otherwise.
    """
    if geom.is_all_point_sky_regions:
        valid = True
    elif isinstance(geom.region, CircleSkyRegion) and rad_max.is_fixed_rad_max:
        valid = np.allclose(geom.region.radius, rad_max.quantity, rtol)

        if not valid:
            raise ValueError(
                f"CircleSkyRegion radius must be equal to RADMAX "
                f"for point-like IRFs with fixed RADMAX. "
                f"Expected {rad_max.quantity} got {geom.region.radius}."
            )
    else:
        valid = False

    return valid


class RegionsFinder(metaclass=ABCMeta):
    """Baseclass for regions finders

    Parameters
    ----------
    binsz : `~astropy.coordinates.Angle`
        Bin size of the reference map used for region finding.
    """

    def __init__(self, binsz=0.01 * u.deg):
        """Create a new RegionFinder"""
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
            reference_geom,
            exclusion_mask,
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
                    excluded = (
                        excluded_pixels.separation(region_test.center) < 1
                    ).any()
                else:
                    excluded = region_test.contains(excluded_pixels).any()

            if not excluded:
                regions.append(region_test)

        # We cannot check for overlap of PointSkyRegion here, this is done later
        # in make_counts_off_rad_max in the rad_max case
        if not isinstance(region, PointSkyRegion):
            if self._are_regions_overlapping(regions, reference_geom):
                log.warning("Found overlapping off regions, returning no regions")
                return [], reference_geom.wcs

        return [r.to_sky(reference_geom.wcs) for r in regions], reference_geom.wcs

    @staticmethod
    def _are_regions_overlapping(regions, reference_geom):
        # check for overlap
        masks = [
            region.to_mask().to_image(reference_geom._shape) > 0 for region in regions
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
        angle_max = FULL_CIRCLE - angle_min - self.min_distance_input
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
                "ReflectedRegionsFinder does not work with PointSkyRegion. Use WobbleRegionsFinder instead."
            )

        regions = []

        reference_geom = self._create_reference_geometry(region, center)
        center_pixel = self._get_center_pixel(center, reference_geom)

        region_pix = self._get_region_pixels(region, reference_geom)
        excluded_pixels = self._get_excluded_pixels(reference_geom, exclusion_mask)

        angle_min, angle_max = self._get_angle_range(
            region=region,
            reference_geom=reference_geom,
            center_pix=center_pixel,
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
    region_finder: RegionsFinder
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
            if kwargs:
                raise ValueError("No kwargs can be given if providing a region_finder")
            self.region_finder = region_finder

    @staticmethod
    def _filter_regions_off_rad_max(regions_off, energy_axis, geom, events, rad_max):
        # check for overlap
        offset = geom.region.center.separation(events.pointing_radec)

        e_min, e_max = energy_axis.bounds
        regions = [geom.region] + regions_off

        overlap = are_regions_overlapping_rad_max(
            regions, rad_max, offset, e_min, e_max
        )

        if overlap:
            log.warning("Found overlapping on/off regions, choose less off regions")
            return []

        return regions_off

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
        geom = dataset.counts.geom
        energy_axis = geom.axes["energy"]
        events = observation.events
        rad_max = observation.rad_max

        is_point_sky_region = geom.is_all_point_sky_regions

        if rad_max and not is_rad_max_compatible_region_geom(
            rad_max=rad_max, geom=geom
        ):
            raise ValueError(
                "Must use `PointSkyRegion` or `CircleSkyRegion` with rad max "
                "equivalent radius in point-like analysis,"
                f" got {type(geom.region)} instead"
            )

        regions_off, wcs = self.region_finder.run(
            center=observation.pointing_radec,
            region=geom.region,
            exclusion_mask=self.exclusion_mask,
        )

        if geom.is_all_point_sky_regions and len(regions_off) > 0:
            regions_off = self._filter_regions_off_rad_max(
                regions_off, energy_axis, geom, events, observation.rad_max
            )

        if len(regions_off) == 0:
            log.warning(
                "ReflectedRegionsBackgroundMaker failed. No OFF region found "
                f"outside exclusion mask for dataset '{dataset.name}'."
            )
            return None, RegionNDMap.from_geom(geom=geom, data=0)

        geom_off = RegionGeom.from_regions(
            regions=regions_off,
            axes=[energy_axis],
            wcs=wcs,
        )

        if is_point_sky_region:
            counts_off = make_counts_off_rad_max(
                geom_off=geom_off,
                rad_max=observation.rad_max,
                events=events,
            )
        else:
            counts_off = RegionNDMap.from_geom(geom=geom_off)
            counts_off.fill_events(events)

        acceptance_off = RegionNDMap.from_geom(geom=geom_off, data=len(regions_off))

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
                f"ReflectedRegionsBackgroundMaker failed. Setting {dataset_onoff.name} "
                "mask to False."
            )
        return dataset_onoff

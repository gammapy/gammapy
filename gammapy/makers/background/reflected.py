# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from regions import PixCoord
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.maps import RegionGeom, RegionNDMap, WcsNDMap, WcsGeom
from ..core import Maker

__all__ = ["ReflectedRegionsFinder", "ReflectedRegionsBackgroundMaker"]

log = logging.getLogger(__name__)


class ReflectedRegionsFinder:
    """Find reflected regions.

    This class is responsible for placing :ref:`region_reflected` for a given
    input region and pointing position. It converts to pixel coordinates
    internally assuming a tangent projection at center position.

    If the center lies inside the input region, no reflected regions
    can be found.

    If you want to make a
    background estimate for an IACT observation using the reflected regions
    method, see also `~gammapy.makers.ReflectedRegionsBackgroundMaker`

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region to rotate
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    angle_increment : `~astropy.coordinates.Angle`, optional
        Rotation angle applied when a region falls in an excluded region.
    min_distance : `~astropy.coordinates.Angle`, optional
        Minimal distance between two consecutive reflected regions
    min_distance_input : `~astropy.coordinates.Angle`, optional
        Minimal distance from input region
    max_region_number : int, optional
        Maximum number of regions to use
    exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
        Exclusion mask
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
    >>> finder = ReflectedRegionsFinder(min_distance_input='1 rad', region=on_region, center=pointing)
    >>> finder.run()
    >>> print(finder.reflected_regions[0])
    Region: CircleSkyRegion
    center: <SkyCoord (ICRS): (ra, dec) in deg
        (83.19879005, 25.57300957)>
    radius: 0.39953342830756855 deg
    """

    def __init__(
        self,
        region,
        center,
        angle_increment="0.1 rad",
        min_distance="0 rad",
        min_distance_input="0.1 rad",
        max_region_number=10000,
        exclusion_mask=None,
        binsz="0.01 deg",
    ):
        self.region = region
        self.center = center

        self.angle_increment = Angle(angle_increment)
        if self.angle_increment <= Angle(0, "deg"):
            raise ValueError("angle_increment is too small")

        self.min_distance = Angle(min_distance)
        self.min_distance_input = Angle(min_distance_input)
        self.exclusion_mask = exclusion_mask
        self.max_region_number = max_region_number
        self.reflected_regions = None
        self.binsz = Angle(binsz)

    @property
    def exclusion_mask_ref(self):
        """Exlcusion mask reprojected"""
        if self.exclusion_mask:
            mask = self.exclusion_mask.interp_to_geom(self.geom_ref)
        else:
            mask = WcsNDMap.from_geom(geom=self.geom_ref, data=1)

        return mask

    @property
    def geom_ref(self):
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
        frame = self.region.center.frame.name

        # width is the full width of an image (not the radius)
        width = 4 * self.region.center.separation(self.center) + Angle("0.3 deg")

        return WcsGeom.create(
            skydir=self.center,
            binsz=self.binsz,
            width=width,
            frame=frame,
            proj="TAN"
        )

    @property
    def region_pix(self):
        """Pixel region"""
        return self.region.to_pixel(self.geom_ref.wcs)

    @property
    def center_pix(self):
        """Center pix coordinate"""
        return PixCoord.from_sky(self.center, self.geom_ref.wcs)

    @property
    def excluded_pix_coords(self):
        """Excluded pix coords"""
        # Extract all pixcoords in the geom
        pix_x, pix_y = self.geom_ref.get_pix()

        # find excluded PixCoords
        mask = self.exclusion_mask_ref.data == 0
        return PixCoord(pix_x[mask], pix_y[mask])

    @property
    def region_angular_size(self):
        """Compute maximum angular size of a group of pixels as seen from center.

        This assumes that the center lies outside the group of pixel

        Returns
        -------
        angular_size : `~astropy.coordinates.Angle`
            the maximum angular size
        """
        mask = self.geom_ref.region_mask([self.region]).data
        pix_x, pix_y = self.geom_ref.get_pix()

        pixels = PixCoord(pix_x[mask], pix_y[mask])

        newX, newY = self.center_pix.x - pixels.x, self.center_pix.y - pixels.y
        angles = Angle(np.arctan2(newX, newY), "rad")
        angular_size = np.max(angles) - np.min(angles)

        if angular_size.value > np.pi:
            angular_size = np.max(angles.wrap_at(0 * u.rad)) - np.min(
                angles.wrap_at(0 * u.rad)
            )

        return angular_size

    @property
    def angle_min(self):
        """Minimum angle"""
        # Minimum angle a region has to be moved to not overlap with previous one
        # Add required minimal distance between two off regions
        return self.region_angular_size + self.min_distance

    @property
    def angle_max(self):
        """"""
        return Angle("360deg") - self.angle_min - self.min_distance_input

    def find_regions(self):
        """Find reflected regions."""
        curr_angle = self.angle_min + self.min_distance_input
        reflected_regions = []

        while curr_angle < self.angle_max:
            test_reg = self.region_pix.rotate(self.center_pix, curr_angle)
            if not np.any(test_reg.contains(self.excluded_pix_coords)):
                region = test_reg.to_sky(self.geom_ref.wcs)
                reflected_regions.append(region)

                curr_angle += self.angle_min
                if self.max_region_number <= len(reflected_regions):
                    break
            else:
                curr_angle = curr_angle + self.angle_increment

        self.reflected_regions = reflected_regions

    def plot(self, fig=None, ax=None):
        """Standard debug plot.

        See example here: :ref:'regions_reflected'.
        """
        fig, ax, cbar = self.reference_map.plot(
            fig=fig, ax=ax, cmap="gray", vmin=0, vmax=1
        )
        wcs = self.reference_map.geom.wcs

        on_patch = self.region.to_pixel(wcs=wcs).as_artist(edgecolor="red", alpha=0.6)
        ax.add_patch(on_patch)

        for off in self.reflected_regions:
            tmp = off.to_pixel(wcs=wcs)
            off_patch = tmp.as_artist(edgecolor="blue", alpha=0.6)
            ax.add_patch(off_patch)

            xx, yy = self.center.to_pixel(wcs)
            ax.plot(xx, yy, marker="+", color="green", markersize=20, linewidth=5)

        return fig, ax


class ReflectedRegionsBackgroundMaker(Maker):
    """Reflected regions background maker.

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
    exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
        Exclusion mask
    binsz : `~astropy.coordinates.Angle`
        Bin size of the reference map used for region finding.
    """

    tag = "ReflectedRegionsBackgroundMaker"

    def __init__(
        self,
        angle_increment="0.1 rad",
        min_distance="0 rad",
        min_distance_input="0.1 rad",
        max_region_number=10000,
        exclusion_mask=None,
        binsz="0.01 deg",
    ):
        self.binsz = binsz
        self.exclusion_mask = exclusion_mask
        self.angle_increment = Angle(angle_increment)
        self.min_distance = Angle(min_distance)
        self.min_distance_input = Angle(min_distance_input)
        self.max_region_number = max_region_number

    def _get_finder(self, dataset, observation):
        return ReflectedRegionsFinder(
            binsz=self.binsz,
            exclusion_mask=self.exclusion_mask,
            center=observation.pointing_radec,
            region=dataset.counts.geom.region,
            min_distance=self.min_distance,
            min_distance_input=self.min_distance_input,
            max_region_number=self.max_region_number,
            angle_increment=self.angle_increment,
        )

    def make_counts_off(self, dataset, observation):
        """Make off counts.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset.
        observation : `DatastoreObservation`
            Data store observation.


        Returns
        -------
        counts_off : `RegionNDMap`
            Off counts.
        """
        finder = self._get_finder(dataset, observation)
        finder.run()

        energy_axis = dataset.counts.geom.axes["energy"]

        if len(finder.reflected_regions) > 0:
            geom = RegionGeom.from_regions(
                regions=finder.reflected_regions,
                axes=[energy_axis],
                wcs=finder.reference_map.geom.wcs
            )

            counts_off = RegionNDMap.from_geom(geom=geom)
            counts_off.fill_events(observation.events)
            acceptance_off = RegionNDMap.from_geom(geom=geom, data=len(finder.reflected_regions))
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
            name=dataset.name
        )

        if dataset_onoff.counts_off is None:
            dataset_onoff.mask_safe.data[...] = False
            log.warning(
                f"ReflectedRegionsBackgroundMaker failed. Setting {dataset_onoff.name} mask to False."
            )
        return dataset_onoff

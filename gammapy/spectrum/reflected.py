# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from regions import PixCoord
from gammapy.maps import WcsNDMap
from gammapy.maps.geom import frame_to_coordsys
from gammapy.utils.regions import compound_region_to_list, list_to_compound_region
from .core import CountsSpectrum
from .dataset import SpectrumDatasetOnOff

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
    method, see also `~gammapy.spectrum.ReflectedRegionsBackgroundMaker`

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
    >>> from gammapy.spectrum import ReflectedRegionsFinder
    >>> pointing = SkyCoord(83.2, 22.7, unit='deg', frame='icrs')
    >>> target_position = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
    >>> theta = Angle(0.4, 'deg')
    >>> on_region = CircleSkyRegion(target_position, theta)
    >>> finder = ReflectedRegionsFinder(min_distance_input='1 rad', region=on_region, center=pointing)
    >>> finder.run()
    >>> print(finder.reflected_regions[0])
    Region: CircleSkyRegion
    center: <SkyCoord (Galactic): (l, b) in deg
        ( 184.9367087, -8.37920222)>
        radius: 0.400147197682 deg
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
        self.reference_map = None
        self.binsz = Angle(binsz)

    def run(self):
        """Run all steps.
        """
        self.reference_map = self.make_reference_map(
            self.region, self.center, self.binsz
        )
        if self.exclusion_mask is not None:
            coords = self.reference_map.geom.get_coord()
            vals = self.exclusion_mask.get_by_coord(coords)
            self.reference_map.data += vals
        else:
            self.reference_map.data += 1

        # Check if center is contained in region
        if self.region.contains(self.center, self.reference_map.geom.wcs):
            self.reflected_regions = []
        else:
            self.setup()
            self.find_regions()

    @staticmethod
    def make_reference_map(region, center, binsz="0.01 deg", min_width="0.3 deg"):
        """Create empty reference map.

        The size of the map is chosen such that all reflected regions are
        contained on the image.
        To do so, the reference map width is taken to be 4 times the distance between
        the target region center and the rotation point. This distance is larger than
        the typical dimension of the region itself (otherwise the rotation point would
        lie inside the region). A minimal width value is added by default in case the
        region center and the rotation center are too close.

        The WCS of the map is the TAN projection at the `center` in the coordinate
        system used by the `region` center.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to rotate
        center : `~astropy.coordinates.SkyCoord`
            Rotation point
        binsz : `~astropy.coordinates.Angle`
            Reference map bin size.
        min_width : `~astropy.coordinates.Angle`
            Minimal map width.

        Returns
        -------
        reference_map : `~gammapy.maps.WcsNDMap`
            Map containing the region
        """
        coordsys = frame_to_coordsys(region.center.frame.name)

        # width is the full width of an image (not the radius)
        width = 4 * region.center.separation(center) + Angle(min_width)

        return WcsNDMap.create(
            skydir=center, binsz=binsz, width=width, coordsys=coordsys, proj="TAN"
        )

    @staticmethod
    def _region_angular_size(pixels, center):
        """Compute maximum angular size of a group of pixels as seen from center.

        This assumes that the center lies outside the group of pixel

        Parameters
        ----------
        pixels : `~astropy.regions.PixCoord`
            the pixels coordinates
        center : `~astropy.regions.PixCoord`
            the center coordinate in pixels

        Returns
        -------
        angular_size : `~astropy.coordinates.Angle`
            the maximum angular size
        """
        newX, newY = center.x - pixels.x, center.y - pixels.y
        angles = Angle(np.arctan2(newX, newY), "rad")
        angular_size = np.max(angles) - np.min(angles)

        if angular_size.value > np.pi:
            angular_size = np.max(angles.wrap_at(0 * u.rad)) - np.min(
                angles.wrap_at(0 * u.rad)
            )

        return angular_size

    def setup(self):
        """Compute parameters for reflected regions algorithm."""
        geom = self.reference_map.geom
        self._pix_region = self.region.to_pixel(geom.wcs)
        self._pix_center = PixCoord.from_sky(self.center, geom.wcs)

        # Make the ON reference map
        mask = geom.region_mask([self.region], inside=True)
        # on_reference_map = WcsNDMap(geom=geom, data=mask)

        # Extract all pixcoords in the geom
        X, Y = geom.get_pix()
        ONpixels = PixCoord(X[mask], Y[mask])

        # find excluded PixCoords
        mask = self.reference_map.data == 0
        self.excluded_pixcoords = PixCoord(X[mask], Y[mask])

        # Minimum angle a region has to be moved to not overlap with previous one
        min_ang = self._region_angular_size(ONpixels, self._pix_center)

        # Add required minimal distance between two off regions
        self._min_ang = min_ang + self.min_distance

        # Maximum possible angle before regions is reached again
        self._max_angle = Angle("360deg") - self._min_ang - self.min_distance_input

    def find_regions(self):
        """Find reflected regions."""
        curr_angle = self._min_ang + self.min_distance_input
        reflected_regions = []

        while curr_angle < self._max_angle:
            test_reg = self._pix_region.rotate(self._pix_center, curr_angle)
            if not np.any(test_reg.contains(self.excluded_pixcoords)):
                region = test_reg.to_sky(self.reference_map.geom.wcs)
                reflected_regions.append(region)

                curr_angle += self._min_ang
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


class ReflectedRegionsBackgroundMaker:
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
            region=dataset.counts.region,
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
        counts_off : `CountsSpectrum`
            Off counts.
        """
        finder = self._get_finder(dataset, observation)
        finder.run()

        region_union = list_to_compound_region(finder.reflected_regions)

        wcs = finder.reference_map.geom.wcs
        events_off = observation.events.select_region(region_union, wcs)

        edges = dataset.counts.energy.edges
        counts_off = CountsSpectrum(
            energy_hi=edges[1:], energy_lo=edges[:-1], region=region_union
        )
        counts_off.fill_events(events_off)
        return counts_off

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
        counts_off = self.make_counts_off(dataset, observation)
        acceptance_off = len(compound_region_to_list(counts_off.region))

        return SpectrumDatasetOnOff(
            counts=dataset.counts,
            counts_off=counts_off,
            gti=dataset.gti,
            name=dataset.name,
            livetime=dataset.livetime,
            edisp=dataset.edisp,
            aeff=dataset.aeff,
            acceptance=1,
            acceptance_off=acceptance_off,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
        )

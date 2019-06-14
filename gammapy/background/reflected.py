# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from regions import PixCoord
from ..maps import WcsNDMap, WcsGeom, Map
from .background_estimate import BackgroundEstimate

__all__ = ["ReflectedRegionsFinder", "ReflectedRegionsBackgroundEstimator"]

log = logging.getLogger(__name__)

# TODO: remove once copy() is supported by `~astropy.regions`
def _region_copy(region):
    """Returns a region copy"""
    # The function copy.deepcopy does not work for these regions classes
    regdict = {_: getattr(region, _) for _ in region._repr_params}
    new_region = region.__class__(region.center, **regdict)
    return new_region

# TODO: remove once rotate() is supported by `~astropy.regions`
def _rotate_pix_region(pix_region, pix_center, angle):
    """Returns rotated source with given angle"""
    new_region = _region_copy(pix_region)

    dx = pix_region.center.x - pix_center.x
    dy = pix_region.center.y - pix_center.y

    # Offset of region in pix coordinates
    offset = np.hypot(dx, dy)
    # initial angle w.r.t. north
    initial_angle = Angle(np.arctan2(dy, dx), "rad")

    x = pix_center.x + offset * np.cos(angle + initial_angle)
    y = pix_center.y + offset * np.sin(angle + initial_angle)
    new_region.center = PixCoord(x=x, y=y)

    if hasattr(new_region, 'angle'):
        region_angle = angle + pix_region.angle.to('rad')
        new_region.angle = region_angle

    return new_region


class ReflectedRegionsFinder:
    """Find reflected regions.

    This class is responsible for placing :ref:`region_reflected` for a given
    input region and pointing position. It converts to pixel coordinates
    internally. At the moment it works only for circles. If you want to make a
    background estimate for an IACT observation using the reflected regions
    method, see also `~gammapy.background.ReflectedRegionsBackgroundEstimator`

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region to rotate
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    angle_increment : `~astropy.coordinates.Angle`, optional
        Rotation angle applied when a region falls in an excluded region.
    min_distance : `~astropy.coordinates.Angle`, optional
        Minimal distance between to reflected regions
    min_distance_input : `~astropy.coordinates.Angle`, optional
        Minimal distance from input region
    max_region_number : int, optional
        Maximum number of regions to use
    exclusion_mask : `~gammapy.maps.WcsNDMap`, optional
        Exclusion mask
    binsz : `~astropy.coordinates.Angle`
        Bin size of the reference map used for region finding. Default : 0.01 deg

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord, Angle
    >>> from regions import CircleSkyRegion
    >>> from gammapy.background import ReflectedRegionsFinder
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
        self.setup()
        self.find_regions()


    @staticmethod
    def make_reference_map(region, center, binsz="0.01 deg"):
        """Create empty reference map.

        The size of the mask is chosen such that all reflected region are
        contained on the image.
        To do so, the reference map width is taken to be 4 times the distance between
        the target region center and the rotation point. This distance is larger than
        the typical dimension of the region itself (otherwise the rotation point would
        lie inside the region).

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region to rotate
        center : `~astropy.coordinates.SkyCoord`
            Rotation point
        binsz : `~astropy.coordinates.Angle`
            Reference map bin size. Default : 0.01 deg

        Returns
        -------
        reference_map : `~gammapy.maps.WcsNDMap`
            Map containing the region
        """

        try:
            reg_center = region.center
        except:
            raise TypeError("Algorithm not yet adapted to this Region shape")

        # width is the full width of an image (not the radius)
        width = 4 * reg_center.separation(center)

        if 'ra' in reg_center.representation_component_names:
            maskmap = WcsNDMap.create(
                skydir=center, binsz=binsz, width=width, coordsys="CEL", proj="TAN"
             )
        else:
            maskmap = WcsNDMap.create(
                skydir=center, binsz=binsz, width=width, coordsys="GAL", proj="TAN"
            )

        return maskmap

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
        angles = Angle(np.arctan2(newX, newY), 'rad')
        angular_size = np.max(angles) - np.min(angles)

        if angular_size.value > np.pi:
            angular_size = np.max(angles.wrap_at(0 * u.rad)) - np.min(angles.wrap_at(0 * u.rad))

        return angular_size

    def setup(self):
        """Compute parameters for reflected regions algorithm."""
        geom = self.reference_map.geom
        self._pix_region = self.region.to_pixel(geom.wcs)
        self._pix_center = PixCoord.from_sky(self.center, geom.wcs)

        if self._pix_region.contains(self._pix_center):
            raise ValueError("Pointing position within the ON region")

        # Make the ON reference map
        mask = geom.region_mask([self.region], inside=True)
        self.on_reference_map = WcsNDMap(geom=geom, data=mask)

        # Extract all pixcoords in the geom
        X,Y = geom.get_pix()
        ONpixels = PixCoord(X[mask], Y[mask])

        mask_array = np.where(self.reference_map.data < 0.5)
        self.excluded_pixcoords = PixCoord(X[mask_array],Y[mask_array])

        # Minimum angle a region has to be moved to not overlap with previous one
        min_ang = self._region_angular_size(ONpixels, self._pix_center)

        # Add required minimal distance between two off regions
        self._min_ang = min_ang + self.min_distance

        # Maximum possible angle before regions is reached again
        self._max_angle = (
             Angle("360deg") - self._min_ang - self.min_distance_input
        )

        # TODO: remove or change to a proper error
        if self._min_ang < 0:
            log.warn("ISSUE self._min_ang=", self._min_ang)

    def find_regions(self):
        """Find reflected regions."""
        curr_angle = self._min_ang + self.min_distance_input
        reflected_regions = []

        while curr_angle < self._max_angle:
            test_reg = _rotate_pix_region(self._pix_region, self._pix_center, curr_angle)

            if not np.any(test_reg.contains(self.excluded_pixcoords)):
                region = test_reg.to_sky(self.reference_map.geom.wcs)
                reflected_regions.append(region)

                curr_angle += self._min_ang
                if self.max_region_number <= len(reflected_regions):
                    break
            else:
                curr_angle = curr_angle + self.angle_increment

        self.reflected_regions = reflected_regions

        # Make the OFF reference map
        mask = self.reference_map.geom.region_mask(self.reflected_regions, inside=True)
        self.off_reference_map = WcsNDMap(geom=self.reference_map.geom, data=mask)

    def plot(self, fig=None, ax=None):
        """Standard debug plot.

        See example here: :ref:'regions_reflected'.
        """
        fig, ax, cbar = self.reference_map.plot(fig=fig, ax=ax, cmap="gray")
        wcs = self.reference_map.geom.wcs

        on_patch = self.region.to_pixel(wcs=wcs).as_artist(color="red", alpha=0.6)
        ax.add_patch(on_patch)

        for off in self.reflected_regions:
            tmp = off.to_pixel(wcs=wcs)
            off_patch = tmp.as_artist(color="blue", alpha=0.6)
            ax.add_patch(off_patch)

            xx, yy = self.center.to_pixel(wcs)
            ax.plot(
                xx, yy,
                marker="+",
                color="green",
                markersize=20,
                linewidth=5,
            )

        return fig, ax


class ReflectedRegionsBackgroundEstimator:
    """Reflected Regions background estimator.

    This class is responsible for creating a
    `~gammapy.background.BackgroundEstimate` by placing reflected regions given
    a target region and an observation.

    For a usage example see :gp-notebook:`spectrum_analysis`

    TODO: make the reference map (to make the mask used to select events)
    Parameters
    ----------
    on_region : `~regions.SkyRegion`
        Target region with any shape, except `~region.PolygonSkyRegion`
    observations : `~gammapy.data.Observations`
        Observations to process
    binsz : `~astropy.coordinates.Angle`
        Optional, bin size of the maps used to compute the regions, Default '0.01 deg'
    kwargs : dict
        Forwarded to `~gammapy.background.ReflectedRegionsFinder`
    """

    def __init__(self, on_region, observations, binsz=0.01*u.deg, **kwargs):
        self.on_region = on_region
        self.observations = observations

        self.binsz = binsz

        self.finder = ReflectedRegionsFinder(region=on_region, center=None, binsz=Angle(self.binsz), **kwargs)

        self.exclusion_mask = kwargs.get("exclusion_mask")
        self.result = None

    def __str__(self):
        s = self.__class__.__name__
        s += "\n{}".format(self.on_region)
        s += "\n{}".format(self.observations)
        s += "\n{}".format(self.finder)
        return s

    def run(self):
        """Run all steps."""
        log.debug("Computing reflected regions")
        result = []
        for obs in self.observations:
            temp = self.process(obs)
            result.append(temp)

        self.result = result

    def process(self, obs):
        """Estimate background for one observation."""
        log.debug("Processing observation {}".format(obs))
        self.finder.center = obs.pointing_radec

        try:
            self.finder.run()
            off_region = self.finder.reflected_regions
            off_events = obs.events.select_map_mask(self.finder.off_reference_map)
            a_off = len(off_region)
            log.info("Found {0} reflected regions for the Obs #{1}".format(a_off, obs.obs_id))

        except ValueError:
            log.warning("Obs #{} rejected! Pointing position within the ON region".format(obs.obs_id))
            off_region = []
            off_events = []
            a_off = 0

        on_events = obs.events.select_map_mask(self.finder.on_reference_map)
        a_on = 1

        return BackgroundEstimate(
            on_region=self.on_region,
            on_events=on_events,
            off_region=off_region,
            off_events=off_events,
            a_on=a_on,
            a_off=a_off,
            method="Reflected Regions",
        )

    def plot(self, fig=None, ax=None, cmap=None, idx=None, add_legend=False):
        """Standard debug plot.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            Top level container of the figure
        ax : `~matplotlib.axes.Axes`
            Axes of the figure
        cmap : `~matplotlib.colors.ListedColormap`, optional
            Color map to use
        idx : int, optional
            Observations to include in the plot, default: all
        add_legend : boolean, optional
            Enable/disable legend in the plot, default: False
        """
        import matplotlib.pyplot as plt

        try:
            reg_center = self.on_region.center
        except:
            raise NotImplementedError("Algorithm not yet adapted to this Region shape")

        if 'ra' in reg_center.representation_component_names:
            coordsys = 'CEL'
        else:
            coordsys = 'GAL'

        pnt_radec = SkyCoord([_.pointing_radec for _ in self.observations])
        width = np.max(5*self.on_region.center.separation(pnt_radec).to_value('deg'))

        geom = WcsGeom.create(skydir=reg_center, binsz=self.binsz, width=width, coordsys=coordsys, proj="TAN")
        plot_map = Map.from_geom(geom)

        if fig is None:
            fig = plt.figure(figsize=(7, 7))

        if self.exclusion_mask is not None:
            coords = geom.get_coord()
            vals = self.exclusion_mask.get_by_coord(coords)
            plot_map.data += vals
        else:
            plot_map.data += 1.

        fig, ax, cbar = plot_map.plot(fig=fig, ax=ax)

        on_patch = self.on_region.to_pixel(wcs=geom.wcs).as_artist(edgecolor="red")
        ax.add_patch(on_patch)

        result = self.result
        obs_list = list(self.observations)
        if idx is not None:
            obs_list = np.asarray(self.observations)[idx]
            obs_list = np.atleast_1d(obs_list)
            result = np.asarray(self.result)[idx]
            result = np.atleast_1d(result)

        cmap = cmap or plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(self.observations)))

        handles = []
        for idx_ in np.arange(len(obs_list)):
            obs = obs_list[idx_]

            off_regions = result[idx_].off_region
            for off in off_regions:
                off_patch = off.to_pixel(wcs=geom.wcs).as_artist(
                    alpha=0.8, edgecolor=colors[idx_], label="Obs {}".format(obs.obs_id)
                )
                handle = ax.add_patch(off_patch)
            if off_regions:
                handles.append(handle)

            xx, yy = obs.pointing_radec.to_pixel(geom.wcs)
            ax.plot(
                xx, yy,
                marker="+",
                color=colors[idx_],
                markersize=20,
                linewidth=5,
            )

        if add_legend:
            ax.legend(handles=handles)

        return fig, ax, cbar

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from regions import PixCoord, CirclePixelRegion, CircleAnnulusPixelRegion, \
                     RectanglePixelRegion, RectangleAnnulusPixelRegion, \
                     EllipsePixelRegion,EllipseAnnulusPixelRegion
from ..maps import WcsNDMap
from .background_estimate import BackgroundEstimate

__all__ = ["ReflectedRegionsFinder", "ReflectedRegionsBackgroundEstimator"]

log = logging.getLogger(__name__)


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
        binsz="0.02 deg",
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

    def run(self, obs_id=-1):
        """Run all steps.
        """
        self.reference_map = self.make_reference_map(
            self.region, self.center, self.binsz
        )
        if self.exclusion_mask is not None:
            _run_exclusion_mask = self.exclusion_mask.reproject(self.reference_map.geom)
            _run_exclusion_mask.data[np.where(np.isnan(_run_exclusion_mask.data))] = 1
            coords = self.reference_map.geom.get_coord()
            vals = _run_exclusion_mask.get_by_coord(coords)
            self.reference_map.data = vals
        else:
            self.reference_map.data = np.ones(self.reference_map.data.shape)
        self.setup()
        self.find_regions()

    @staticmethod
    def make_reference_map(region, center, binsz):
        """Create empty reference map.

        The size of the mask is chosen such that all reflected region are
        contained on the image.

        Returns
        -------
        reference_map : `~gammapy.maps.WcsNDMap`
            Map containing the region
        """


        try:
            reg_center = region.center
        except:
            raise NotImplementedError("Algorithm not yet adapted to this Region shape")

        # width is the full width of an image (not the radius)
        width = Angle(3.0 * reg_center.transform_to(center).separation(center), u.degree) * 2.

        if 'ra' in reg_center.representation_component_names:
            maskmap = WcsNDMap.create(
                skydir=center, binsz=binsz, width=width, coordsys="CEL", proj="TAN"
             )
        else:
            maskmap = WcsNDMap.create(
                skydir=self.center, binsz=self.binsz, width=width, coordsys="GAL", proj="TAN"
            )

        return maskmap

    def setup(self):
        """Compute parameters for reflected regions algorithm."""
        geom = self.reference_map.geom
        self._pix_region = self.region.to_pixel(geom.wcs)
        self._pix_center = PixCoord.from_sky(self.center, geom.wcs)
        dx = self._pix_region.center.x - self._pix_center.x
        dy = self._pix_region.center.y - self._pix_center.y

        if self._pix_region.contains(self._pix_center):
            log.warn("Obs #{} rejected! Pointing position within the ON region".format(self.obs_id))
            return

        # Offset of region in pix coordinates
        self._offset = np.hypot(dx, dy)

        # Make the ON reference map
        _mask = geom.region_mask([self.region], inside=True)
        self.on_reference_map = WcsNDMap(geom=geom, data=_mask)
        # TODO: remove this lign after the correction of the Issue #2074
        self.on_reference_map.data = self.on_reference_map.data.astype(float)

        # Starting angle of region
        self._angle = Angle(np.arctan2(dy, dx), "rad")

        # Minimum angle a circle has to be moved to not overlap with previous one
        pix_idx = geom.get_pix()
        pix_on_x = pix_idx[0][_mask]
        pix_on_y = pix_idx[1][_mask]
        newX, newY = (self._pix_center.x - pix_on_x), (self._pix_center.y - pix_on_y)
        angles = np.arctan2(newY, newX) * u.rad
        min_ang = np.max(angles)-np.min(angles)
        if min_ang.value > np.pi:
            new_angles = np.zeros_like(angles)
            new_angles[angles.value > 0] = angles[angles.value > 0] - np.pi*u.rad
            new_angles[angles.value < 0] = angles[angles.value < 0] + np.pi*u.rad
            min_ang = np.max(new_angles)-np.min(new_angles)

        # Add required minimal distance between two off regions
        self._min_ang = min_ang + self.min_distance

        # Maximum possible angle before regions is reached again
        self._max_angle = (
            self._angle + Angle("360deg") - self._min_ang - self.min_distance_input
        )

        if self._min_ang < 0:
            print("ISSUE self._min_ang=", self._min_ang)
        if self.min_distance_input < 0:
            print("ISSUE self.min_distance_input=", self.min_distance_input)

    def find_regions(self):
        """Find reflected regions."""
        curr_angle = self._angle + self._min_ang + self.min_distance_input
        reflected_regions = []
        geom = self.reference_map.geom
        _run_exclusion_mask = None
        _pixel_excluded = False
        if not self.exclusion_mask is None:
            _run_exclusion_mask = self.exclusion_mask.reproject(self.reference_map.geom)
            _run_exclusion_mask.data[np.where(np.isnan(_run_exclusion_mask.data))] = 1
            _mask_array = np.where(_run_exclusion_mask.data < 0.99)
            _excluded_coords = geom.get_coord().apply_mask(_mask_array).to_coordsys('CEL')
            if len(_excluded_coords[0]) > 0 and len(_excluded_coords[1]) > 0 :
                _excluded_pixcoords = PixCoord(*SkyCoord(*_excluded_coords, unit='deg').to_pixel(geom.wcs))
                _pixel_excluded = True

        while curr_angle < self._max_angle:
            _test_reg = self._create_rotated_reg(curr_angle)
            if _pixel_excluded is False or not np.any(_test_reg.contains(_excluded_pixcoords)):
                _region = _test_reg.to_sky(geom.wcs)
                log.debug("Placing reflected region\n{}".format(_region))
                reflected_regions.append(_region)
                curr_angle = curr_angle + self._min_ang
                if self.max_region_number <= len(reflected_regions):
                    break
            else:
                curr_angle = curr_angle + self.angle_increment

        print("Found {0} reflected regions for the Obs #{1}".format(len(reflected_regions), self.obs_id))
        log.debug("Found {} reflected regions".format(len(reflected_regions)))

        self.reflected_regions = reflected_regions

        # Make the OFF reference map
        _mask = geom.region_mask(self.reflected_regions, inside=True)
        self.off_reference_map = WcsNDMap(geom=geom, data=_mask)
        # TODO: remove this lign after the correction of the Issue #2074
        self.off_reference_map.data = self.off_reference_map.data.astype(float)

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

    @staticmethod
    def _compute_xy(pix_center, offset, angle):
        """Compute x, y position for a given position angle and offset."""
        dx = offset * np.cos(angle)
        dy = offset * np.sin(angle)
        x = pix_center.x + dx
        y = pix_center.y + dy
        return PixCoord(x=x, y=y)

    def _create_rotated_reg(self, curr_angle):
        """ Compute a rotated region"""
        _test_pos = self._compute_xy(self._pix_center, self._offset, curr_angle)

        # The function copy.deepcopy does not work for these regions classes
        dict = { _ : getattr(self._pix_region, _) for _ in self._pix_region._repr_params}
        _test_reg = self._pix_region.__class__(_test_pos, **dict)

        if hasattr(_test_reg, 'angle'):
            _angle = curr_angle - self._angle + self._pix_region.angle.to('rad')
            _test_reg.angle = _angle

        return _test_reg


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
    kwargs : dict
        Forwarded to `~gammapy.background.ReflectedRegionsFinder`
    """

    def __init__(self, on_region, observations, **kwargs):
        self.on_region = on_region
        self.observations = observations

        # Proper management of the bin size (as it has impact on results)
        binsz = 0.01 * u.deg
        if "binsz" in kwargs:
            binsz = kwargs.get("binsz")
        else:
            binsz = self._get_bounding_size(on_region, observations[0].pointing_radec)/10.
            if binsz > 0.01*u.deg:
                binsz = 0.01*u.deg
        self.binsz = binsz
        log.debug("ReflectedRegionsFinder Ref. Map: bins={}".format(self.binsz))
        self.finder = ReflectedRegionsFinder(region=on_region, center=None, binsz=Angle(binsz), **kwargs)

        self.exclusion_mask = kwargs.get("exclusion_mask")
        self.result = None

    def __str__(self):
        s = self.__class__.__name__
        s += "\n{}".format(self.on_region)
        s += "\n{}".format(self.observations)
        s += "\n{}".format(self.finder)
        return s

    @staticmethod
    def _get_bounding_size(region, center):
        """Return the typical radius of the bounding size"""

        _map = WcsNDMap.create(
            skydir=center, binsz=Angle("0.01 deg"), width=10., coordsys="GAL", proj="TAN"
        )
        wcs = _map.geom.wcs

        region_pix = region.to_pixel(wcs)
        _ixmin = region_pix.bounding_box.ixmin
        _ixmax = region_pix.bounding_box.ixmax
        _iymin = region_pix.bounding_box.iymin
        _iymax = region_pix.bounding_box.iymax
        _min_point = PixCoord(_ixmin, _iymin).to_sky(wcs)
        _max_point = PixCoord(_ixmax, _iymax).to_sky(wcs)

        return np.round(_min_point.separation(_max_point), decimals=3)

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
        self.finder.run(obs.obs_id)
        off_region = self.finder.reflected_regions
        off_events = obs.events.select_map_mask(self.finder.off_reference_map)
        on_events = obs.events.select_map_mask(self.finder.on_reference_map)
        a_on = 1
        a_off = len(off_region)
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
            _plotmap = WcsNDMap.create(
                skydir=reg_center, binsz=self.binsz, width=10., coordsys="CEL", proj="TAN"
             )
        else:
            _plotmap = WcsNDMap.create(
                skydir=reg_center, binsz=self.binsz, width=10., coordsys="GAL", proj="TAN"
            )
        _plotmap.data = np.ones(_plotmap.data.shape)
        wcs = _plotmap.geom.wcs

        if fig is None:
            fig = plt.figure(figsize=(7, 7))
        fig, ax, cbar = _plotmap.plot(fig=fig, ax=ax)
        if not (self.exclusion_mask is None):
            self.exclusion_mask.reproject(_plotmap.geom).plot(fig=fig, ax=ax, cmap="gray")

        wcs = self.finder.reference_map.geom.wcs
        on_patch = self.on_region.to_pixel(wcs=wcs).as_artist(edgecolor="red")
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
                off_patch = off.to_pixel(wcs=wcs).as_artist(
                    alpha=0.8, edgecolor=colors[idx_], label="Obs {}".format(obs.obs_id)
                )
                handle = ax.add_patch(off_patch)
            if off_regions:
                handles.append(handle)

            xx, yy = obs.pointing_radec.to_pixel(wcs)
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

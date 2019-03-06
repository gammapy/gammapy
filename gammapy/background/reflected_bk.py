# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from scipy.ndimage import distance_transform_edt
from astropy.coordinates import Angle
from astropy import units as u
from regions import PixCoord, CirclePixelRegion
from ..maps import WcsNDMap
from .background_estimate import BackgroundEstimate

__all__ = ["ReflectedRegionsFinder_BK", "ReflectedRegionsBackgroundEstimator_BK"]

log = logging.getLogger(__name__)


def _compute_distance_image_BK(mask_map):
    """Distance to nearest exclusion region.

    Compute distance image, i.e. the Euclidean (=Cartesian 2D)
    distance (in pixels) to the nearest exclusion region.

    We need to call distance_transform_edt twice because it only computes
    dist for pixels outside exclusion regions, so to get the
    distances for pixels inside we call it on the inverted mask
    and then combine both distance images into one, using negative
    distances (note the minus sign) for pixels inside exclusion regions.

    If data consist only of ones, it'll be supposed to be far away
    from zero pixels, so in capacity of answer it should be return
    the matrix with the shape as like as data but packed by constant
    value Max_Value (MAX_VALUE = 1e10).

    If data consist only of zeros, it'll be supposed to be deep inside
    an exclusion region, so in capacity of answer it should be return
    the matrix with the shape as like as data but packed by constant
    value -Max_Value (MAX_VALUE = 1e10).

    Returns
    -------
    distance : `~gammapy.maps.WcsNDMap`
        Map of distance to nearest exclusion region.
    """
    max_value = 1e10

    if np.all(mask_map.data == 1):
        dist_map = mask_map.copy(data=mask_map.data * max_value)
        return dist_map

    if np.all(mask_map.data == 0):
        dist_map = mask_map.copy(data=mask_map.data - max_value)
        return dist_map

    distance_outside = distance_transform_edt(mask_map.data)

    invert_mask = np.invert(np.array(mask_map.data, dtype=np.bool))
    distance_inside = distance_transform_edt(invert_mask)

    distance = np.where(
        mask_map.data,
        distance_outside,
        -distance_inside,  # pylint:disable=invalid-unary-operand-type
    )

    return mask_map.copy(data=distance)


class ReflectedRegionsFinder_BK:
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
        binsz=Angle(0.01 *u.deg)
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

    def run(self, obs_id):
        """Run all steps.
        """
        self.obs_id = obs_id
        self.reference_map = self.make_reference_map(self)
        if self.exclusion_mask is not None:
            coords = self.reference_map.geom.get_coord()
            vals = self.exclusion_mask.get_by_coord(coords)
            self.reference_map.data += vals
        else:
            self.reference_map.data += 1
        self.setup()
        self.find_regions()

    @staticmethod
    def make_reference_map(self):
        """Create empty reference map.

        The size of the mask is chosen such that all reflected region are
        contained on the image.

        Returns
        -------
        reference_map : `~gammapy.maps.WcsNDMap`
            Map containing the region
        """
        self.width = 15.

        try:
            reg_center = self.region.center
        except:
            raise NotImplementedError("Algorithm not yet adapted to this Region shape")

        if 'ra' in reg_center.representation_component_names:
            _maskmap = WcsNDMap.create(
                skydir=self.center, binsz=self.binsz, width=self.width, coordsys="CEL", proj="TAN"
             )
        else:
            _maskmap = WcsNDMap.create(
                skydir=self.center, binsz=self.binsz, width=self.width, coordsys="GAL", proj="TAN"
            )

        # wcs = _maskmap.geom.wcs
        # _region_pix = self.region.to_pixel(wcs)
        # _ixmin = _region_pix.bounding_box.ixmin
        # _ixmax = _region_pix.bounding_box.ixmax
        # _iymin = _region_pix.bounding_box.iymin
        # _iymax = _region_pix.bounding_box.iymax
        # self.cog = PixCoord((_ixmin + _ixmax) / 2., (_iymin + _iymax) / 2.).to_sky(wcs)
        self.width = Angle(3.0 * reg_center.transform_to(self.center).separation(self.center), u.degree)
        maskmap = _maskmap.cutout(self.center, self.width)

        return maskmap

    def setup(self):
        """Compute parameters for reflected regions algorithm."""
        geom = self.reference_map.geom
        self._pix_region = self.region.to_pixel(geom.wcs)
        self._pix_center = PixCoord(*self.center.to_pixel(geom.wcs))
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

        # Starting angle of region
        self._angle = Angle(np.arctan2(dx, dy), "rad")

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

        # Distance image
        # self._distance_image = _compute_distance_image_BK(self.reference_map)

    def find_regions(self):
        """Find reflected regions."""
        curr_angle = self._angle + self._min_ang + self.min_distance_input
        reflected_regions = []
        while curr_angle < self._max_angle:
            test_pos = self._compute_xy(self._pix_center, self._offset, curr_angle)
            # TODO : to generalise to any shape
            test_reg = CirclePixelRegion(test_pos, self._pix_region.radius)
            _region = test_reg.to_sky(self.exclusion_mask.geom.wcs)
            if not self._is_inside_exclusion(_region):
                refl_region = test_reg.to_sky(self.reference_map.geom.wcs)
                log.debug("Placing reflected region\n{}".format(refl_region))
                reflected_regions.append(refl_region)
                curr_angle = curr_angle + self._min_ang
                if self.max_region_number <= len(reflected_regions):
                    break
            else:
                curr_angle = curr_angle + self.angle_increment

        log.debug("Found {} reflected regions".format(len(reflected_regions)))

        self.reflected_regions = reflected_regions

    def plot(self, fig=None, ax=None):
        """Standard debug plot.

        TODO : if a center is defined for the on_region, then make the plot in the WCS of this center (e.g.: ICRS for AGN)

        See example here: :ref:'regions_reflected'.
        """
        fig, ax, cbar = self.reference_map.plot(fig=fig, ax=ax, cmap="gray")
        wcs = self.reference_map.geom.wcs
        # self.on_reference_map.plot(fig=fig, ax=ax)

        # This is good
        on_patch = self.region.to_pixel(wcs=wcs).as_artist(color="red", alpha=0.6)
        ax.add_patch(on_patch)

        for off in self.reflected_regions:
            tmp = off.to_pixel(wcs=wcs)
            off_patch = tmp.as_artist(color="blue", alpha=0.6)
            ax.add_patch(off_patch)

            test_pointing = self.center
            ax.scatter(
                test_pointing.galactic.l.degree,
                test_pointing.galactic.b.degree,
                transform=ax.get_transform("galactic"),
                marker="+",
                s=300,
                linewidths=3,
                color="green",
            )

        return fig, ax

    def _is_inside_exclusion(self, testreg):
        """Test if a `~regions.SkyRegion` overlaps with an exclusion mask.

        If the regions is outside the exclusion mask (0 when to be excluded), return 'False'
        """
        if self.exclusion_mask is None:
            return False

        geom = self.exclusion_mask.geom
        _mask = geom.region_mask([testreg], inside=True)
        pix_idx = geom.get_pix()
        pix_on_x = pix_idx[0][_mask]
        pix_on_y = pix_idx[1][_mask]

        for x, y in zip(pix_on_x, pix_on_y):
            try:
                val = self.exclusion_mask.data[np.round(y).astype(int), np.round(x).astype(int)]
                # print("[{0}, {1}] val={2}".format(x, y, val))
            except IndexError:
                continue
            if val is False:
                return True

        return False

    @staticmethod
    def _compute_xy(pix_center, offset, angle):
        """Compute x, y position for a given position angle and offset.

        # TODO: replace by calculation using `astropy.coordinates`
        """
        dx = offset * np.sin(angle)
        dy = offset * np.cos(angle)
        x = pix_center.x + dx
        y = pix_center.y + dy
        return PixCoord(x=x, y=y)


class ReflectedRegionsBackgroundEstimator_BK:
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
        self.finder = ReflectedRegionsFinder_BK(region=on_region, center=None, binsz=Angle(binsz), **kwargs)

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
        # off_events = obs.events.select_circular_region(off_region)      # TODO: replace with the Regis PR
        # on_events = obs.events.select_circular_region(self.on_region)   # TODO: replace with the Regis PR
        off_events = obs.events
        on_events = obs.events
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

        IsGal = True
        if 'ra' in reg_center.representation_component_names:
            _plotmap = WcsNDMap.create(
                skydir=reg_center, binsz=self.binsz, width=10., coordsys="CEL", proj="TAN"
             )
            IsGal = False
        else:
            _plotmap = WcsNDMap.create(
                skydir=reg_center, binsz=self.binsz, width=10., coordsys="GAL", proj="TAN"
            )

        fig, ax, cbar = _plotmap.plot(fig=fig, ax=ax)
        # if not (self.finder.exclusion_mask is None):
        #     self.finder.exclusion_mask.plot(fig=fig, ax=ax, cmap="gray")

        wcs = _plotmap.geom.wcs
        on_patch = self.on_region.to_pixel(wcs=wcs).as_artist(color="red")
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
                    alpha=0.8, color=colors[idx_], label="Obs {}".format(obs.obs_id)
                )
                handle = ax.add_patch(off_patch)
            if off_regions:
                handles.append(handle)

            if IsGal:
                test_pointing = obs.pointing_radec.galactic
                ax.scatter(
                    test_pointing.l.degree,
                    test_pointing.b.degree,
                    transform=ax.get_transform("galactic"),
                    marker="+",
                    color=colors[idx_],
                    s=300,
                    linewidths=3,
                )
            else:
                test_pointing = obs.pointing_radec
                ax.scatter(
                    test_pointing.ra.degree,
                    test_pointing.dec.degree,
                    marker="+",
                    color=colors[idx_],
                    s=300,
                    linewidths=3,
                )
                print("BKH> OK {0} {1}".format(test_pointing.ra.degree,test_pointing.dec.degree))

        if add_legend:
            ax.legend(handles=handles)

        return fig, ax, cbar

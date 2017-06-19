# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from regions import PixCoord, CirclePixelRegion
from ..image import SkyImage
from .background_estimate import BackgroundEstimate
import logging

__all__ = [
    'ReflectedRegionsFinder',
    'ReflectedRegionsBackgroundEstimator',
]

log = logging.getLogger(__name__)


class ReflectedRegionsFinder(object):
    """Find reflected regions.

    This class is responsible for placing :ref:`region_reflected` for a given
    input region and pointing position. It converts to pixel coordinates
    internally. At the moment it works only for circles. If you want to make a
    background estimate for an IACT observation using the reflected regions
    method, see also `~gammapy.background.ReflectedRegionsBackgroundEstimator`

    Parameters
    ----------
    region : `~regions.CircleSkyRegion`
        Region to rotate
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    angle_increment : `~astropy.coordinates.Angle`, optional
        Rotation angle for each step
    min_distance : `~astropy.coordinates.Angle`, optional
        Minimal distance between to reflected regions
    min_distance_input : `~astropy.coordinates.Angle`, optional
        Minimal distance from input region
    exclusion_mask : `~gammapy.image.SkyImage`, optional
        Exclusion mask

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord, Angle
    >>> from regions import CircleSkyRegion
    >>> from gammapy.background import ReflectedRegionsFinder
    >>> pointing = SkyCoord(83.2, 22.7, unit='deg', frame='icrs')
    >>> target_position = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
    >>> theta = Angle(0.4, 'deg')
    >>> on_region = CircleSkyRegion(target_position, theta)
    >>> finder = ReflectedRegionsFinder(min_distance_input='1 rad', region=on_region, center=pointing))
    >>> regions = finder.run()
    >>> print(regions[0])
    Region: CircleSkyRegion
    center: <SkyCoord (Galactic): (l, b) in deg
        ( 184.9367087, -8.37920222)>
        radius: 0.400147197682 deg
    """

    def __init__(self, region, center,
                 angle_increment='0.1 rad', min_distance='0 rad',
                 min_distance_input='0.1 rad', exclusion_mask=None):
        self.region = region
        self.center = center
        self.angle_increment = Angle(angle_increment)
        self.min_distance = Angle(min_distance)
        self.min_distance_input = Angle(min_distance_input)
        self.exclusion_mask = exclusion_mask

        self.reflected_regions = None

    def run(self):
        """Run all steps.
        """
        if self.exclusion_mask is None:
            self.exclusion_mask = self.make_empty_mask(self.region, self.center)
        self.setup()
        self.find_regions()

    @staticmethod
    def make_empty_mask(region, center):
        """Create empty exclusion mask.

        The size of the mask is chosen such that all reflected region are
        contained on the image.

        Parameters
        ----------
        region : `~regions.CircleSkyRegion`
            Region to rotate
        center : `~astropy.coordinates.SkyCoord`
            Rotation point
        """
        log.debug('No exclusion mask provided, creating an emtpy one')
        min_size = region.center.separation(center)
        binsz = 0.02
        npix = int((3 * min_size / binsz).value)
        exclusion_mask = SkyImage.empty(name='empty exclusion mask',
                                        xref=center.galactic.l.value,
                                        yref=center.galactic.b.value,
                                        binsz=binsz,
                                        nxpix=npix,
                                        nypix=npix,
                                        fill=1)
        return exclusion_mask

    def setup(self):
        """Compute parameters for reflected regions algorithm."""
        wcs = self.exclusion_mask.wcs
        self._pix_region = self.region.to_pixel(wcs)
        self._pix_center = PixCoord(*self.center.to_pixel(wcs))
        dx = self._pix_region.center.x - self._pix_center.x
        dy = self._pix_region.center.y - self._pix_center.y

        # Offset of region in pix coordinates
        self._offset = np.hypot(dx, dy)

        # Starting angle of region
        self._angle = Angle(np.arctan2(dx, dy), 'rad')

        # Minimum angle a circle has to be moved to not overlap with previous one
        min_ang = Angle(2 * np.arcsin(self._pix_region.radius / self._offset), 'rad')

        # Add required minimal distance between two off regions
        self._min_ang = min_ang + self.min_distance

        # Maximum possible angle before regions is reached again
        self._max_angle = self._angle + Angle('360deg') - self._min_ang - self.min_distance_input

        # Distance image
        self._distance_image = self.exclusion_mask.distance_image

    def find_regions(self):
        """Find reflected regions."""
        curr_angle = self._angle + self._min_ang + self.min_distance_input
        reflected_regions = []
        while curr_angle < self._max_angle:
            test_pos = self._compute_xy(self._pix_center, self._offset, curr_angle)
            test_reg = CirclePixelRegion(test_pos, self._pix_region.radius)
            if not self._is_inside_exclusion(test_reg, self._distance_image):
                refl_region = test_reg.to_sky(self.exclusion_mask.wcs)
                log.debug('Placing reflected region\n{}'.format(refl_region))
                reflected_regions.append(refl_region)
                curr_angle = curr_angle + self._min_ang
            else:
                curr_angle = curr_angle + self.angle_increment

        log.info('Found {} reflected regions'.format(len(reflected_regions)))
        self.reflected_regions = reflected_regions

    def plot(self, fig=None, ax=None):
        """Standard debug plot.

        See example here: :ref:'regions_reflected'.
        """
        fig, ax, cbar = self.exclusion_mask.plot(fig=fig, ax=ax, cmap='gray')
        wcs = self.exclusion_mask.wcs
        on_patch = self.region.to_pixel(wcs=wcs).as_patch(color='red', alpha=0.6)
        ax.add_patch(on_patch)

        for off in self.reflected_regions:
            tmp = off.to_pixel(wcs=wcs)
            off_patch = tmp.as_patch(color='blue', alpha=0.6)
            ax.add_patch(off_patch)

            test_pointing = self.center
            ax.scatter(test_pointing.galactic.l.degree, test_pointing.galactic.b.degree,
                       transform=ax.get_transform('galactic'),
                       marker='+', s=300, linewidths=3, color='green')

        return fig, ax

    @staticmethod
    def _is_inside_exclusion(pixreg, distance_image):
        """Test if a `~regions.PixRegion` overlaps with an exclusion mask.

        If the regions is outside the exclusion mask, return 'False'
        """
        x, y = pixreg.center.x, pixreg.center.y
        try:
            val = distance_image.data[np.round(y).astype(int),
                                      np.round(x).astype(int)]
        except IndexError:
            return False
        else:
            return val < pixreg.radius

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


class ReflectedRegionsBackgroundEstimator(object):
    """Reflected Regions background estimator.

    This class is responsible for creating a
    `~gammapy.background.BackgroundEstimate` by placing reflected regions given
    a target region and an observation.

    For a usage example see :gp-extra-notebook:`spectrum_analysis`

    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        Target region
    obs_list : `~gammapy.data.ObservationList`
        Observations to process
    kwargs : dict
        Forwarded to `gammapy.background.ReflectedRegionsFinder`
    """

    def __init__(self, on_region, obs_list, **kwargs):
        self.on_region = on_region
        self.obs_list = obs_list
        self.finder = ReflectedRegionsFinder(region=on_region,
                                             center=None,
                                             **kwargs)

        self.result = None

    def __str__(self):
        """String representation of the class."""
        s = self.__class__.__name__
        s += '\n{}'.format(self.on_region)
        s += '\n{}'.format(self.obs_list)
        s += '\n{}'.format(self.finder)
        return s

    def run(self):
        """Run all steps."""
        log.info('Running {}'.format(self))
        result = []
        for obs in self.obs_list:
            temp = self.process(obs=obs)
            result.append(temp)

        self.result = result

    def process(self, obs):
        """Estimate background for one observation."""
        log.info('Processing observation {}'.format(obs))
        self.finder.center = obs.pointing_radec
        self.finder.run()
        off_region = self.finder.reflected_regions
        off_events = obs.events.select_circular_region(off_region)
        on_events = obs.events.select_circular_region(self.on_region)
        a_on = 1
        a_off = len(off_region)
        return BackgroundEstimate(on_region=self.on_region,
                                  on_events=on_events,
                                  off_region=off_region,
                                  off_events=off_events,
                                  a_on=a_on,
                                  a_off=a_off,
                                  method='Reflected Regions')

    def plot(self, fig=None, ax=None, cmap=None, idx=None):
        """Standard debug plot.

        Parameters
        ----------
        cmap : `~matplotlib.colors.ListedColormap`, optional
            Color map to use
        idx : int, optional
            Observations to include in the plot, default: all
        """
        import matplotlib.pyplot as plt

        fig, ax, cbar = self.finder.exclusion_mask.plot(fig=fig, ax=ax)

        wcs = self.finder.exclusion_mask.wcs
        on_patch = self.on_region.to_pixel(wcs=wcs).as_patch(color='red')
        ax.add_patch(on_patch)

        result = self.result
        obs_list = self.obs_list
        if idx is not None:
            obs_list = np.asarray(obs_list)[idx]
            obs_list = np.atleast_1d(obs_list)
            result = np.asarray(self.result)[idx]
            result = np.atleast_1d(result)

        cmap = cmap or plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(self.obs_list)))

        handles = []
        for idx_ in np.arange(len(obs_list)):
            obs = obs_list[idx_]

            off_regions = result[idx_].off_region
            for off in off_regions:
                off_patch = off.to_pixel(wcs=wcs).as_patch(
                    alpha=0.8, color=colors[idx_],
                    label='Obs {}'.format(obs.obs_id),
                )
                handle = ax.add_patch(off_patch)
            if off_regions:
                handles.append(handle)

            test_pointing = obs.pointing_radec.galactic
            ax.scatter(
                test_pointing.l.degree, test_pointing.b.degree,
                transform=ax.get_transform('galactic'),
                marker='+', color=colors[idx_], s=300, linewidths=3,
            )

        ax.legend(handles=handles)

        return fig, ax, cbar

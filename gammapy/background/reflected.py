# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from regions import PixCoord, CirclePixelRegion
from ..image import SkyImage
from .background_estimate import BackgroundEstimate

__all__ = [
    'find_reflected_regions',
    'ReflectedRegionsBackgroundEstimator',
]


def find_reflected_regions(region, center, exclusion_mask=None,
                           angle_increment='0.1 rad',
                           min_distance='0 rad',
                           min_distance_input='0.1 rad'):
    """Find reflected regions.

    Converts to pixel coordinates internally.

    Parameters
    ----------
    region : `~regions.CircleSkyRegion`
        Region
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    exclusion_mask : `~gammapy.image.SkyImage`, optional
        Exclusion mask
    angle_increment : `~astropy.coordinates.Angle`
        Rotation angle for each step
    min_distance : `~astropy.coordinates.Angle`
        Minimal distance between to reflected regions
    min_distance_input : `~astropy.coordinates.Angle`
        Minimal distance from input region

    Returns
    -------
    regions : list of `~regions.SkyRegion`
        Reflected regions list
    """
    angle_increment = Angle(angle_increment)
    min_distance = Angle(min_distance)
    min_distance_input = Angle(min_distance_input)

    # Create empty exclusion mask if None is provided
    if exclusion_mask is None:
        exclusion_mask = make_default_exclusion_mask(center, exclusion_mask, region)

    distance_image = exclusion_mask.distance_image

    reflected_regions_pix = list()
    wcs = exclusion_mask.wcs
    pix_region = region.to_pixel(wcs)
    pix_center = PixCoord(*center.to_pixel(wcs))

    # Compute angle of the ON regions
    dx = pix_region.center.x - pix_center.x
    dy = pix_region.center.y - pix_center.y
    offset = np.hypot(dx, dy)
    angle = Angle(np.arctan2(dx, dy), 'rad')

    # Get the minimum angle a Circle has to be moved in order to not overlap
    # with the previous one
    min_ang = Angle(2 * np.arcsin(pix_region.radius / offset), 'rad')

    # Add required minimal distance between two off regions
    min_ang += min_distance

    # Maximum allowed angle before the an overlap with the ON regions happens
    max_angle = angle + Angle('360 deg') - min_ang - min_distance_input

    # Starting angle
    curr_angle = angle + min_ang + min_distance_input

    while curr_angle < max_angle:
        test_pos = _compute_xy(pix_center, offset, curr_angle)
        test_reg = CirclePixelRegion(test_pos, pix_region.radius)
        if distance_image.lookup_pix(test_reg.center) > pix_region.radius:
            reflected_regions_pix.append(test_reg)
            curr_angle = curr_angle + min_ang
        else:
            curr_angle = curr_angle + angle_increment

    reflected_regions = [_.to_sky(wcs) for _ in reflected_regions_pix]
    return reflected_regions


def make_default_exclusion_mask(center, region):
    min_size = region.center.separation(center)
    binsz = 0.02
    npix = int((3 * min_size / binsz).value)
    return SkyImage.empty(
        name='empty exclusion mask',
        xref=center.galactic.l.value,
        yref=center.galactic.b.value,
        binsz=binsz,
        nxpix=npix,
        nypix=npix,
        fill=1,
    )


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

    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        Target region
    obs_list : `~gammapy.data.ObservationList`
        List of observations to process
    exclusion : `~gammapy.image.SkyImage`
        Exclusion mask
    config : dict
        Config dict to be passed to :func:`gammapy.background.find_reflected_regions`
    """

    def __init__(self, on_region, obs_list, exclusion, config=dict()):
        self.on_region = on_region
        self.obs_list = obs_list
        self.exclusion = exclusion
        self.result = None
        self.config = config

    def __str__(self):
        """String representation of the class."""
        s = self.__class__.__name__ + '\n'
        s += str(self.on_region)
        s += '\n'.format(self.config)
        return s

    @staticmethod
    def process(on_region, obs, exclusion, **kwargs):
        """Estimate background for one observation.

        kwargs are forwaded to :func:`gammapy.background.find_reflected_regions`

        Parameters
        ----------
        on_region : `~regions.CircleSkyRegion`
            Target region
        obs : `~gammapy.data.DataStoreObservation`
            Observation
        exclusion : `~gammapy.image.ExclusionMask`
            ExclusionMask

        Returns
        -------
        background : `~gammapy.background.BackgroundEstimate`
            Reflected regions background estimate
        """
        off_region = find_reflected_regions(region=on_region,
                                            center=obs.pointing_radec,
                                            exclusion_mask=exclusion,
                                            **kwargs)
        # TODO: Properly use regions package
        off_events = obs.events.select_circular_region(off_region)
        a_on = 1
        a_off = len(off_region)
        return BackgroundEstimate(off_region, off_events, a_on, a_off, tag='reflected')

    def run(self):
        """Process all observations."""
        result = []
        for obs in self.obs_list:
            temp = self.process(on_region=self.on_region,
                                obs=obs,
                                exclusion=self.exclusion,
                                **self.config)
            result.append(temp)

        self.result = result

    def plot(self, fig=None, ax=None, cmap=None, idx=None):
        """Debug plot

        Parameters
        ----------
        cmap : `~matplotlib.colors.ListedColormap`, optional
            Color map to use
        idx : int, optional
            Observations to include in the plot, default: all
        """
        import matplotlib.pyplot as plt

        fig, ax, cbar = self.exclusion.plot(fig=fig, ax=ax)

        on_patch = self.on_region.to_pixel(wcs=self.exclusion.wcs).as_patch(color='red')
        ax.add_patch(on_patch)

        if idx is None:
            obs_list = self.obs_list
            result = self.result
        else:
            obs_list = np.asarray(self.obs_list)[idx]
            obs_list = np.atleast_1d(obs_list)
            result = np.asarray(self.result)[idx]
            result = np.atleast_1d(result)

        handles = list()
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(obs_list)))
        for idx_ in np.arange(len(obs_list)):
            obs = obs_list[idx_]
            for off in result[idx_].off_region:
                tmp = off.to_pixel(wcs=self.exclusion.wcs)
                off_patch = tmp.as_patch(alpha=0.8, color=colors[idx_],
                                         label='Obs {}'.format(obs.obs_id))
                handle = ax.add_patch(off_patch)
            handles.append(handle)

            test_pointing = obs.pointing_radec
            ax.scatter(test_pointing.galactic.l.degree, test_pointing.galactic.b.degree,
                       transform=ax.get_transform('galactic'),
                       marker='+', color=colors[idx_], s=300, linewidths=3)

        ax.legend(handles=handles)

        return fig, ax

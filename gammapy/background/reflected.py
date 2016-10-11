# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from regions import PixCoord, CirclePixelRegion
from ..image import SkyMask
from .background_estimate import BackgroundEstimate
import logging

__all__ = [
    'find_reflected_regions',
    'ReflectedRegionsBackgroundEstimator',
]


log = logging.getLogger(__name__)

def find_reflected_regions(region, center, exclusion_mask=None,
                           angle_increment=None, min_distance=None,
                           min_distance_input=None):
    """Find reflected regions.

    Converts to pixel coordinates internally.

    Parameters
    ----------
    region : `~regions.CircleSkyRegion`
        Region
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    exclusion_mask : `~gammapy.image.SkyMask`, optional
        Exclusion mask
    angle_increment : `~astropy.coordinates.Angle`
        Rotation angle for each step, default: 0.1 rad
    min_distance : `~astropy.coordinates.Angle`
        Minimal distance between to reflected regions, default: 0 rad
    min_distance_input : `~astropy.coordinates.Angle`
        Minimal distance from input region, default: 0.1 rad

    Returns
    -------
    regions : list of `~regions.SkyRegion`
        Reflected regions list
    """
    angle_increment = Angle('0.1 rad') if angle_increment is None else Angle(angle_increment)
    min_distance = Angle('0 rad') if min_distance is None else Angle(min_distance)
    min_distance_input = Angle('0.1 rad') if min_distance_input is None else Angle(min_distance_input)

    # Create empty exclusion mask if None is provided
    if exclusion_mask is None:
        min_size = region.center.separation(center)
        binsz = 0.02
        npix = int((3 * min_size / binsz).value)
        exclusion_mask = SkyMask.empty(name='empty exclusion mask',
                                      xref=center.galactic.l.value,
                                      yref=center.galactic.b.value,
                                      binsz=binsz,
                                      nxpix=npix,
                                      nypix=npix,
                                      fill=1)

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

    # Maximum allowd angle before the an overlap with the ON regions happens
    max_angle = angle + Angle('360deg') - min_ang - min_distance_input

    # Starting angle
    curr_angle = angle + min_ang + min_distance_input
    while curr_angle < max_angle:
        test_pos = _compute_xy(pix_center, offset, curr_angle)
        test_reg = CirclePixelRegion(test_pos, pix_region.radius)
        if not _is_inside_exclusion(test_reg, exclusion_mask):
            reflected_regions_pix.append(test_reg)
            curr_angle = curr_angle + min_ang
        else:
            curr_angle = curr_angle + angle_increment

    reflected_regions = [_.to_sky(wcs) for _ in reflected_regions_pix]
    log.debug('Found {} reflected regions:\n {}'.format(len(reflected_regions),
                                                            reflected_regions))
    return reflected_regions


def _compute_xy(pix_center, offset, angle):
    """Compute x, y position for a given position angle and offset

    # TODO: replace by calculation using `astropy.coordinates`
    """
    dx = offset * np.sin(angle)
    dy = offset * np.cos(angle)
    x = pix_center.x + dx
    y = pix_center.y + dy
    return PixCoord(x=x, y=y)


# TODO :Copied from gammapy.region.PixCircleList (deleted), find better place
def _is_inside_exclusion(pixreg, exclusion):
    x, y = pixreg.center.x, pixreg.center.y
    image = exclusion.distance_image
    excl_dist = image.data
    val = excl_dist[np.round(y).astype(int), np.round(x).astype(int)]
    return val < pixreg.radius


class ReflectedRegionsBackgroundEstimator(object):
    """Reflected Regions background estimator
    
    Parameters
    ----------
    on_region : `~regions.CircleSkyRegion`
        Target region
    obs_list : `~gammapy.data.ObservationList`
        List of observations to process
    exclusion : `~gammapy.image.SkyMask`
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
        s = self.__class__.__name__ + '\n'
        s += str(self.on_region)
        s += '\n'.format(self.config)
        return s

    @staticmethod
    def process(on_region, obs, exclusion, **kwargs):
        """Estimate background for one observation

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
        off_region = find_reflected_regions(region = on_region,
                                            center = obs.pointing_radec,
                                            exclusion_mask=exclusion,
                                            **kwargs)
        # TODO: Properly use regions package
        off_events = obs.events.select_circular_region(off_region)
        a_on = 1
        a_off = len(off_region)
        return BackgroundEstimate(off_region, off_events, a_on, a_off, tag='reflected')

    def run(self):
        """Process all observations"""
        result = []
        for obs in self.obs_list:
            temp = self.process(on_region = self.on_region,
                                obs = obs,
                                exclusion = self.exclusion,
                                **self.config)
            result.append(temp)

        self.result = result

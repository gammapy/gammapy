# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reflected region background estimation.
"""
from __future__ import print_function, division
import numpy as np
from numpy import sin, cos, arctan2, sqrt
from ..image import exclusion_distance, lookup

__all__ = ['ReflectedRegionMaker', 'ReflectedBgMaker']


class ReflectedRegionMaker(object):
    """Finds reflected regions.

    TODO: At the moment only works for circular regions!

    TODO: should work with world or pixel coordinates internally!???
    """

    def __init__(self, exclusion, fov, min_on_distance=0.1, angle_increment=0.1):
        self.exclusion = exclusion
        self.fov = fov
        self.min_on_distance = min_on_distance
        self.regions = None
        self.angle_increment = angle_increment

        # For a "distance to exclusion region" map.
        # This will allow a fast check if a given position
        # is a valid off region position simply by looking
        # up the distance of the corresponding pixel in this map.
        header = exclusion.header
        CDELT = exclusion.header['CDELT2']
        distance = CDELT * exclusion_distance(exclusion.data)
        from astropy.io.fits import ImageHDU
        self.exclusion_distance = ImageHDU(distance, header)

    def compute(self, x_on, y_on, r_on):
        self.regions = []
        self.on_region = dict(x=x_on, y=y_on, r=r_on)
        angle_on = self._compute_angle(x_on, y_on)
        offset_on = self._compute_offset(x_on, y_on)
        angles = angle_on + np.arange(0, np.radians(360), np.radians(self.angle_increment))
        for angle in angles:
            x, y = self._compute_xy(offset_on, angle)
            if self._is_position_ok(x, y, r_on):
                region = dict(x=x, y=y, r=r_on)
                self.regions.append(region)

    def _is_position_ok(self, x, y, r):
        if self._is_exclusion_ok(x, y, r):
            if self._is_other_regions_ok(x, y, r):
                return True
        return False

    def _is_exclusion_ok(self, x, y, r):
        return lookup(self.exclusion_distance, x, y) > r

    def _is_other_regions_ok(self, x, y, r):
        other_regions = self.regions + [self.on_region]
        for region in other_regions:
            x2 = (region['x'] - x) ** 2
            y2 = (region['y'] - y) ** 2
            distance = sqrt(x2 + y2)
            min_distance = r + region['r']
            if distance < min_distance:
                return False
        return True

    def _compute_offset(self, x, y):
        x2 = (x - self.fov['x']) ** 2
        y2 = (y - self.fov['y']) ** 2
        return sqrt(x2 + y2)

    def _compute_angle(self, x, y):
        """Compute position angle wrt. the FOV center"""
        dx = x - self.fov['x']
        dy = y - self.fov['y']
        angle = arctan2(dx, dy)
        return angle

    def _compute_xy(self, offset, angle):
        """Compute x, y position for a given position angle"""
        dx = offset * sin(angle)
        dy = offset * cos(angle)
        x = self.fov['x'] + dx
        y = self.fov['y'] + dy
        return x, y

    def write_off_regions(self, filename):
        fmt = 'galactic; circle({x},{y},{r})\n'
        with open(filename, 'w') as fh:
            for region in self.regions:
                line = fmt.format(**region)
                fh.write(line)


class ReflectedBgMaker(object):
    """Compute background using the reflected background method"""
    pass

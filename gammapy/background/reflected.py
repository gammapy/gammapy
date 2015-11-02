# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reflected region background estimation.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy import sin, cos, arctan2, sqrt
from ..image import exclusion_distance, lookup
from astropy.table import Table

__all__ = [
    'ReflectedRegionMaker',
    'CircularOffRegions',
    'find_reflected_regions_for_event_list'
]


class CircularOffRegions(Table):
    """List of circular OFF regions

    Parameters
    ----------
    data_rows : `~np.array`
        Array containing the circle definitions: x, y, radius
    """

    @property
    def number_of_regions(self):
        """Number of OFF regions"""
        return len(self)
    
    def info(self):
        """Summary info string."""
        s = super(CircularOffRegions, self).__repr__()
        return s

    def to_ds9(self, filename):
        """Write ds9 regions file"""

        fmt = 'fk5; circle({x},{y},{r})\n'
        with open(filename, 'w') as fh:
            for row in self:
                x = row['x']
                y = row['y']
                r = row['r']
                line = fmt.format(x=x,y=y,r=r)
                fh.write(line)


class ReflectedRegionMaker(object):
    """Finds reflected regions.

    More info on the reflected regions background estimation methond
    can be found in [Berge2007]_

    TODO: At the moment only works for circular regions!
    TODO: should work with world or pixel coordinates internally!???

    Parameters
    ----------
    exclusion : ImageHDU
        Excluded regions mask
    fov : `~astropy.coordinates.SkyCoord`
        Center of the field of view (pointing)
    angle_increment : float (optional)
        Angle between two reflected regions
    """

    def __init__(self, exclusion, fov, min_on_distance=0.1, angle_increment=0.1):
        self.exclusion = exclusion
        self.fov = {'x' : fov.ra.value , 'y' : fov.dec.value}
        self.min_on_distance = min_on_distance
        self.regions = None
        self.angle_increment = angle_increment

        # For a "distance to exclusion region" map.
        # This will allow a fast check if a given position
        # is a valid off region position simply by looking
        # up the distance of the corresponding pixel in this map.
        header = exclusion.header
        CDELT = exclusion.header['CDELT2']
        excl_mask = np.array(exclusion.data, dtype = 'int')
        distance = CDELT * exclusion_distance(excl_mask)
        from astropy.io.fits import ImageHDU
        self.exclusion_distance = ImageHDU(distance, header)

    def compute(self, target, r_on):
        """Computes reflected regions for a given (circular) On region

        Parameters
        ----------
        target : `~astropy.coordinates.SkyCoord`
            Center of the ON region
        r_on : `~astropy.coordinates.Angle`
            Radius of the ON region

        Returns
        -------
        off_regions : `~gammapy.background.CircularOffRegions`
            Table containing the Off regions
        """

        self.regions = []
        x_on, y_on, r_on = target.ra.value, target.dec.value, r_on.to('deg').value
        self.on_region = dict(x=x_on, y=y_on, r=r_on)
        angle_on = self._compute_angle(x_on, y_on)
        offset_on = self._compute_offset(x_on, y_on)
        angles = angle_on + np.arange(0, np.radians(360), np.radians(self.angle_increment))
        for angle in angles:
            x, y = self._compute_xy(offset_on, angle)
            if self._is_position_ok(x, y, r_on):
                region = dict(x=x, y=y, r=r_on)
                self.regions.append(region)

        return CircularOffRegions(rows=self.regions, meta={'frame':'fk5'})

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


def find_reflected_regions_for_event_list(event_list, exclusion,
                                          on_radius, target=None):
    """Find reflected regions for a given event list

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
        Event List
    exclusion : ImageHDU
        Excluded regions mask
    on_radius : `~astropy.coordinates.Angle`
        Size of the On region
    target : `~astropy.coordinates.SkyCoord`, optional
        Source position, taken from event list header if not specified
    
    Returns
    -------
    off_regions : `~gammapy.background.CircularOffRegions`
        Table containing the Off regions
    
    Examples
    --------
    .. code-block:: python

        from gammapy.datasets import gammapy_extra
        from gammapy.data import EventList 
        from astropy.io import fits

        exclfile = gammapy_extra.filename('
             test_datasets/spectrum/dummy_exclusion.fits')
        eventfile = gammapy_extra.filename('
             test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')

        exclusion = fits.open(exclfile, hdu = 0)[0]
        event_list = EventList.read(eventfile, hdu='EVENTS') 
        on_rad = Angle(0.3, 'deg')

        regions = find_reflected_regions_for_event_list(
             event_list, exclusion, on_rad)

    """
    
    fov = event_list.pointing_radec
    maker = ReflectedRegionMaker(exclusion, fov)
    if target is None:
        target = event_list.target_radec
    regions = maker.compute(target, on_radius)
    return regions
    
        

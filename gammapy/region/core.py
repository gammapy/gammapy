# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
from astropy.extern import six

__all__ = [
    'Region',
    'PixRegion',
    'PixRegionList',
    'SkyRegion',
    'SkyRegionList',
]


@six.add_metaclass(abc.ABCMeta)
class Region(object):
    """Base class for all regions.
    """

    def intersection(self, other):
        """Returns a region representing the intersection of this region with ``other``.
        """
        raise NotImplementedError

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        raise NotImplementedError

    def union(self, other):
        """Returns a region representing the union of this region with ``other``.
        """
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class PixRegion(Region):
    """Base class for all regions defined in pixel coordinates.
    """

    def __contains__(self, pixcoord):
        """Checks whether a position or positions fall inside the region.

        Parameters
        ----------
        pixcoord : tuple
            The position or positions to check, as a tuple of scalars or
            arrays. In future this could also be a `PixCoord` instance.
        """
        raise NotImplementedError

    def area(self):
        """Returns the area of the region as a `~astropy.units.Quantity`.
        """
        raise NotImplementedError

    def to_sky(self, wcs):
        """Returns a region defined in sky coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system transformation to assume
        """
        raise NotImplementedError

    def to_mask(self, mode='center'):
        """Returns a mask for the aperture.

        Parameters
        ----------
        mode : { 'center' | 'any' | 'all' | 'exact' }
            The following modes are available:
                * ``'center'``: returns 1 for pixels where the center is in
                  the region, and 0 otherwise.
                * ``'any'``: returns 1 for pixels where any of the pixel is
                  in the region, and 0 otherwise.
                * ``'all'``: returns 1 for pixels that are completely inside
                  the region, 0 otherwise.
                * ``'exact'``: returns a value between 0 and 1 giving the
                  fractional level of overlap of the pixel with the region.

        Returns
        -------
        mask : `~numpy.ndarray`
            A mask indicating whether each pixel is contained in the region.
            slice_x, slice_y : `slice`
            Slices for x and y which can be used on an array to extract the
            same region as the mask.
        """
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class SkyRegion(Region):
    """Base class for all regions defined in celestial coordinates.
    """

    def __contains__(self, skycoord):
        """Checks whether a position or positions fall inside the region.

        Parameters
        ----------
        skycoord : `~astropy.coordinates.SkyCoord`
            The position or positions to check

        Returns
        -------
        contains : bool
            Does this region contain the coordinate?
        """
        raise NotImplementedError

    def area(self):
        """Returns the area of the region as a `~astropy.units.Quantity`.
        """
        raise NotImplementedError

    def to_pixel(self, wcs):
        """Returns a region defined in pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS` instance
            The world coordinate system transformation to assume
        """
        raise NotImplementedError

    def to_dict(self):
        """Create dict that can be used for serialization"""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dict):
        """Create from dict"""
        raise NotImplementedError


class SkyRegionList(list):
    """List of sky regions.
    """

    def to_ds9(self):
        """Convert to ds9 region string.
        """
        ss = ''
        for region in self:
            ss += region.to_ds9()
        return ss

    def write(self, filename, format='ds9'):
        """Write list of regions to file.

        Parameters
        ----------
        filename : str
            Name of file to write
        format : {'ds9'}
            File format
        """

        if format == 'ds9':
            ss = self.to_ds9()
        else:
            raise ValueError('Format {} not definded'.format(format))

        with open(filename, 'w') as fh:
            fh.write(ss)

    def plot(self, ax, **kwargs):
        """Plot all regions in the list"""
        for reg in self:
            reg.plot(ax, **kwargs)

    def to_dict(self):
        """Convert all regions to a list of dicts"""
        out = list()
        for reg in self:
            out.append(reg.to_dict())
        return out

    @classmethod
    def from_dict(cls, inlist):
        """Read a list of regions from list of dicts

        Only Circle regions are supported at the Moment
        """
        from .circle import SkyCircleRegion
        reglist = list()
        for reg in inlist:
            reglist.append(SkyCircleRegion.from_dict(reg))
        return cls(reglist)


class PixRegionList(list):
    """List of pix regions.
    """

    def to_sky(self, wcs, frame='galactic'):
        """Convert to SkyRegions.

        Returns
        -------
        sky_list : `~gammapy.region.SkyRegionList`
            List of SkyRegions
        """
        val = SkyRegionList()
        for region in self:
            sky = region.to_sky(wcs, frame=frame)
            val.append(sky)
        return val

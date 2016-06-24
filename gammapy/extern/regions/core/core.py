# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
from astropy.extern import six
import operator

__all__ = ['Region', 'PixelRegion', 'SkyRegion']


@six.add_metaclass(abc.ABCMeta)
class Region(object):
    """
    Base class for all regions.
    """

    @abc.abstractmethod
    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        raise NotImplementedError("")

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __contains__(self, coord):
        if not coord.isscalar:
            raise ValueError('{} must be scalar'.format(coord))
        return self.contains(coord)


@six.add_metaclass(abc.ABCMeta)
class PixelRegion(Region):
    """
    Base class for all regions defined in pixel coordinates
    """

    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        from .compound import CompoundPixelRegion
        return CompoundPixelRegion(self, other, operator.and_)

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        from .compound import CompoundPixelRegion
        return CompoundPixelRegion(self, other, operator.xor)

    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        from .compound import CompoundPixelRegion
        return CompoundPixelRegion(self, other, operator.or_)

    @abc.abstractmethod
    def contains(self, pixcoord):
        """
        Checks whether a position or positions fall inside the region.

        Parameters
        ----------
        pixcoord : tuple
            The position or positions to check, as a tuple of scalars or
            arrays. In future this could also be a `PixCoord` instance.
        """
        raise NotImplementedError("")

    @abc.abstractproperty
    def area(self):
        """
        Returns the area of the region as a `~astropy.units.Quantity`.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def to_sky(self, wcs, mode='local', tolerance=None):
        """
        Returns a region defined in sky coordinates.

        Parameters
        ----------

        wcs : `~astropy.wcs.WCS` instance
            The world coordinate system transformation to assume

        mode : str
            Convering to sky coordinates can be done with various degrees of
            approximation, which can be set with this option. Possible values
            are:

            * `'local'`: assume that the field of view is small and that
              pixels are square, so that e.g. a circle in sky coordinates
              would be a circle in pixel coordinates. This is the fastest and
              most commonly used for e.g. photometry.

            * `'affine'`: approximate any deviations from the 'local'
              assumption by an affine transformation, e.g. a circle would
              become a rotated ellipse.

            * `'full'`: return an arbitrarily complex polygon in sky
              coordinates that represents the full level of distortion due to
              the conversion from pixel to world coordinates. The degree of
              exactness can be controlled by the ``tolerance`` argument.

        tolerance : `~astropy.units.Quantity`
            The tolerance for the ``'full'`` mode described above.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def to_mask(self, mode='center'):
        """
        Returns a mask for the aperture.

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
        raise NotImplementedError("")

    @abc.abstractmethod
    def to_shapely(self):
        """
        Convert this region to a Shapely object.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def as_patch(self, **kwargs):
        """Convert to mpl patch

        Returns
        -------
        patch : `~matplotlib.patches.Patch`
            Matplotlib patch
        """
        raise NotImplementedError

    def plot(self, ax=None, **kwargs):
        """
        Calls as_patch method forwarding all kwargs and adds patch
        to given axis.

        Parameters
        ----------
        ax : `~matplotlib.axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        patch = self.as_patch(**kwargs)
        ax.add_patch(patch)

        return ax


@six.add_metaclass(abc.ABCMeta)
class SkyRegion(Region):
    """
    Base class for all regions defined in celestial coordinates
    """

    def intersection(self, other):
        """
        Returns a region representing the intersection of this region with
        ``other``.
        """
        from .compound import CompoundSkyRegion
        return CompoundSkyRegion(self, other, operator.and_)

    def symmetric_difference(self, other):
        """
        Returns the union of the two regions minus any areas contained in the
        intersection of the two regions.
        """
        from .compound import CompoundSkyRegion
        return CompoundSkyRegion(self, other, operator.xor)

    def union(self, other):
        """
        Returns a region representing the union of this region with ``other``.
        """
        from .compound import CompoundSkyRegion
        return CompoundSkyRegion(self, other, operator.or_)

    @abc.abstractmethod
    def contains(self, skycoord):
        """
        Checks whether a position or positions fall inside the region.

        Parameters
        ----------
        skycoord : `~astropy.coordinates.SkyCoord`
            The position or positions to check
        """
        raise NotImplementedError("")

    @abc.abstractproperty
    def area(self):
        """
        Returns the area of the region as a `~astropy.units.Quantity`.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def to_pixel(self, wcs, mode='local', tolerance=None):
        """
        Returns a region defined in pixel coordinates.

        Parameters
        ----------

        wcs : `~astropy.wcs.WCS` instance
            The world coordinate system transformation to assume

        mode : str
            Convering to pixel coordinates can be done with various degrees
            of approximation, which can be set with this option. Possible
            values are:

            * `'local'`: assume that the field of view is small and that
              pixels are square, so that e.g. a circle in sky coordinates
              would be a circle in pixel coordinates. This is the fastest and
              most commonly used for e.g. photometry.

            * `'affine'`: approximate any deviations from the 'local'
              assumption by an affine transformation, e.g. a circle would
              become a rotated ellipse.

            * `'full'`: return an arbitrarily complex polygon in pixel
              coordinates that represents the full level of distortion due to
              the conversion from world to pixel coordinates. The degree of
              exactness can be controlled by the ``tolerance`` argument.

        tolerance : `~astropy.units.Quantity`
            The tolerance for the ``'full'`` mode described above.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def as_patch(self, ax, **kwargs):
        """Convert to mpl patch using a given wcs axis

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`
            WCS axis object

        Returns
        -------
        patch : `~matplotlib.patches.Patch`
            Matplotlib patch
        """
        raise NotImplementedError

    def plot(self, ax=None, **kwargs):
        """
        Calls as_patch method forwarding all kwargs and adds patch
        to given wcs axis.

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`
            WCS axis object
        """
        patch = self.as_patch(ax, **kwargs)
        ax.add_patch(patch)

        return ax

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Regions helper functions.

Throughout Gammapy, we use `regions` to represent and work with regions.

https://astropy-regions.readthedocs.io

The functions ``make_region`` and ``make_pixel_region`` should be used
throughout Gammapy in all functions that take ``region`` objects as input.
They do conversion to a standard form, and some validation.

We might add in other conveniences and features here, e.g. sky coord contains
without a WCS (see "sky and pixel regions" in PIG 10), or some HEALPix integration.

TODO: before Gammapy v1.0, discuss what to do about ``gammapy.utils.regions``.
Options: keep as-is, hide from the docs, or to remove it completely
(if the functionality is available in ``astropy-regions`` directly.
"""
from regions import DS9Parser, SkyRegion, PixelRegion, Region, CircleSkyRegion

__all__ = ["make_region", "make_pixel_region"]


def make_region(region):
    """Make region object (`regions.Region`).

    See also:

    * `gammapy.utils.regions.make_pixel_region`
    * https://astropy-regions.readthedocs.io/en/latest/ds9.html
    * http://ds9.si.edu/doc/ref/region.html

    Parameters
    ----------
    region : `regions.Region` or str
        Region object or DS9 string representation

    Examples
    --------
    If a region object in DS9 string format is given, the corresponding
    region object is created. Note that in the DS9 format "image"
    or "physical" coordinates start at 1, whereas `regions.PixCoord`
    starts at 0 (as does Python, Numpy, Astropy, Gammapy, ...).

    >>> from gammapy.utils.regions import make_region
    >>> make_region("image;circle(10,20,3)")
    <CirclePixelRegion(PixCoord(x=9.0, y=19.0), radius=3.0)>
    >>> make_region("galactic;circle(10,20,3)")
    <CircleSkyRegion(<SkyCoord (Galactic): (l, b) in deg
        (10., 20.)>, radius=3.0 deg)>

    If a region object is passed in, it is returned unchanged:

    >>> region = make_region("image;circle(10,20,3)")
    >>> region2 = make_region(region)
    >>> region is region2
    True
    """
    if isinstance(region, str):
        # This is basic and works for simple regions
        # It could be extended to cover more things,
        # like e.g. compound regions, exclusion regions, ....
        return DS9Parser(region).shapes[0].to_region()
    elif isinstance(region, Region):
        return region
    else:
        raise TypeError("Invalid type: {!r}".format(region))


def make_pixel_region(region, wcs=None):
    """Make pixel region object (`regions.PixelRegion`).

    See also: `gammapy.utils.regions.make_region`

    Parameters
    ----------
    region : `regions.Region` or str
        Region object or DS9 string representation
    wcs : `astropy.wcs.WCS`
        WCS

    Examples
    --------
    >>> from gammapy.maps import WcsGeom
    >>> from gammapy.utils.regions import make_pixel_region
    >>> wcs = WcsGeom.create().wcs
    >>> region = make_pixel_region("galactic;circle(10,20,3)", wcs)
    >>> region
    <CirclePixelRegion(PixCoord(x=570.9301128316974, y=159.935542455567), radius=6.061376992149382)>
    """
    if isinstance(region, str):
        region = make_region(region)

    if isinstance(region, SkyRegion):
        if wcs is None:
            raise ValueError("Need wcs to convert to pixel region")
        return region.to_pixel(wcs)
    elif isinstance(region, PixelRegion):
        return region
    else:
        raise TypeError("Invalid type: {!r}".format(region))


class SphericalCircleSkyRegion(CircleSkyRegion):
    """Spherical circle sky region.

    TODO: is this separate class a good idea?

    - If yes, we could move it to the ``regions`` package?
    - If no, we should implement some other solution.
      Probably the alternative is to add extra methods to
      the ``CircleSkyRegion`` class and have that support
      both planar approximation and spherical case?
      Or we go with the approach to always make a
      TAN WCS and not have true cone select at all?
    """

    def contains(self, skycoord, wcs=None):
        """Defined by spherical distance."""
        separation = self.center.separation(skycoord)
        return separation < self.radius

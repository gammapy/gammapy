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
import operator
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import (
    CircleAnnulusSkyRegion,
    CircleSkyRegion,
    CompoundSkyRegion,
    DS9Parser,
    PixelRegion,
    RectangleSkyRegion,
    Region,
    SkyRegion,
)

__all__ = [
    "make_region",
    "make_pixel_region",
    "make_orthogonal_rectangle_sky_regions",
    "make_concentric_annulus_sky_regions",
    "compound_region_to_list",
    "list_to_compound_region",
]


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
        raise TypeError(f"Invalid type: {region!r}")


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
        raise TypeError(f"Invalid type: {region!r}")


def compound_region_to_list(region):
    """Create list of regions from compound regions.

    Parameters
    ----------
    regions : `~regions.CompoundSkyRegion` or `~regions.SkyRegion`
        Compound sky region

    Returns
    -------
    regions : list of `~regions.SkyRegion`
        List of regions.
    """
    regions = []

    if isinstance(region, CompoundSkyRegion):
        if region.operator is operator.or_:
            regions_1 = compound_region_to_list(region.region1)
            regions.extend(regions_1)

            regions_2 = compound_region_to_list(region.region2)
            regions.extend(regions_2)
        else:
            raise ValueError("Only union operator supported")
    else:
        return [region]

    return regions


def list_to_compound_region(regions):
    """Create compound region from list of regions, by creating the union.

    Parameters
    ----------
    regions : list of `~regions.SkyRegion`
        List of regions.

    Returns
    -------
    compound : `~regions.CompoundSkyRegion`
        Compound sky region
    """

    region_union = regions[0]

    for region in regions[1:]:
        region_union = region_union.union(region)

    return region_union


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


def make_orthogonal_rectangle_sky_regions(start_pos, end_pos, wcs, height, nbin=1):
    """Utility returning an array of regions to make orthogonal projections

    Parameters
    ----------
    start_pos : `~astropy.regions.SkyCoord`
        First sky coordinate defining the line to which the orthogonal boxes made
    end_pos : `~astropy.regions.SkyCoord`
        Second sky coordinate defining the line to which the orthogonal boxes made
    height : `~astropy.quantity.Quantity`
        Height of the rectangle region.
    wcs : `~astropy.wcs.WCS`
        WCS projection object
    nbin : int
        Number of boxes along the line

    Returns
    --------
    regions : list of `~regions.RectangleSkyRegion`
        Regions in which the profiles are made
    """
    pix_start = start_pos.to_pixel(wcs)
    pix_stop = end_pos.to_pixel(wcs)

    points = np.linspace(start=pix_start, stop=pix_stop, num=nbin + 1).T
    centers = 0.5 * (points[:, :-1] + points[:, 1:])
    coords = SkyCoord.from_pixel(centers[0], centers[1], wcs)

    width = start_pos.separation(end_pos).to("rad") / nbin
    angle = end_pos.position_angle(start_pos) - 90 * u.deg

    regions = []

    for center in coords:
        reg = RectangleSkyRegion(
            center=center, width=width, height=u.Quantity(height), angle=angle
        )
        regions.append(reg)

    return regions


def make_concentric_annulus_sky_regions(center, radius_max, nbin=11):
    """Make a list of concentric annulus regions.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        Center coordinate
    radius_max : `~astropy.units.Quantity`
        Maximum radius.
    nbin : int
        Number of boxes along the line

    Returns
    -------
    regions : list of `~regions.RectangleSkyRegion`
        Regions in which the profiles are made
    """
    regions = []

    edges = np.linspace(0 * u.deg, u.Quantity(radius_max), nbin)

    for r_in, r_out in zip(edges[:-1], edges[1:]):
        region = CircleAnnulusSkyRegion(
            center=center, inner_radius=r_in, outer_radius=r_out,
        )
        regions.append(region)

    return regions

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Regions helper functions.

Throughout Gammapy, we use `regions` to represent and work with regions.

https://astropy-regions.readthedocs.io

We might add in other conveniences and features here, e.g. sky coord contains
without a WCS (see "sky and pixel regions" in PIG 10), or some HEALPix integration.

TODO: before Gammapy v1.0, discuss what to do about ``gammapy.utils.regions``.
Options: keep as-is, hide from the docs, or to remove it completely
(if the functionality is available in ``astropy-regions`` directly.
"""
import operator
import numpy as np
from scipy.optimize import Bounds, minimize
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import (
    CircleAnnulusSkyRegion,
    CircleSkyRegion,
    CompoundSkyRegion,
    EllipseSkyRegion,
    PolygonSkyRegion,
    RectangleSkyRegion,
    Regions,
)
import matplotlib.pyplot as plt

__all__ = [
    "compound_region_to_regions",
    "make_concentric_annulus_sky_regions",
    "make_orthogonal_rectangle_sky_regions",
    "regions_to_compound_region",
    "region_to_frame",
]


def compound_region_center(compound_region):
    """Compute center for a CompoundRegion

    The center of the compound region is defined here as the geometric median
    of the individual centers of the regions. The geometric median is defined
    as the point the minimises the distance to all other points.

    Parameters
    ----------
    compound_region : `CompoundRegion`
        Compound region

    Returns
    -------
    center : `~astropy.coordinates.SkyCoord`
        Geometric median of the positions of the individual regions
    """
    regions = compound_region_to_regions(compound_region)

    if len(regions) == 1:
        return regions[0].center

    positions = SkyCoord([region.center.icrs for region in regions])

    def f(x, coords):
        """Function to minimize"""
        lon, lat = x
        center = SkyCoord(lon * u.deg, lat * u.deg)
        return np.sum(center.separation(coords).deg)

    ra, dec = positions.ra.wrap_at("180d").deg, positions.dec.deg

    ub = np.array([np.max(ra), np.max(dec)])
    lb = np.array([np.min(ra), np.min(dec)])

    if np.all(ub == lb):
        bounds = None
    else:
        bounds = Bounds(ub=ub, lb=lb)

    result = minimize(
        f,
        x0=[np.mean(ra), np.mean(dec)],
        args=(positions,),
        bounds=bounds,
        method="L-BFGS-B",
    )

    return SkyCoord(result.x[0], result.x[1], frame="icrs", unit="deg")


def compound_region_to_regions(region):
    """Create list of regions from compound regions.

    Parameters
    ----------
    region : `~regions.CompoundSkyRegion` or `~regions.SkyRegion`
        Compound sky region

    Returns
    -------
    regions : `~regions.Regions`
        List of regions.
    """
    regions = Regions([])

    if isinstance(region, CompoundSkyRegion):
        if region.operator is operator.or_:
            regions_1 = compound_region_to_regions(region.region1)
            regions.extend(regions_1)

            regions_2 = compound_region_to_regions(region.region2)
            regions.extend(regions_2)
        else:
            raise ValueError("Only union operator supported")
    else:
        return Regions([region])

    return regions


def regions_to_compound_region(regions):
    """Create compound region from list of regions, by creating the union.

    Parameters
    ----------
    regions : `~regions.Regions`
        List of regions.

    Returns
    -------
    compound : `~regions.CompoundSkyRegion` or `~regions.CompoundPixelRegion`
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

    def contains(integral_map, skycoord, wcs=None):
        """Defined by spherical distance."""
        separation = integral_map.center.separation(skycoord)
        return separation < integral_map.radius


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
    -------
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


def make_concentric_annulus_sky_regions(
    center, radius_max, radius_min=1e-5 * u.deg, nbin=11
):
    """Make a list of concentric annulus regions.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        Center coordinate
    radius_max : `~astropy.units.Quantity`
        Maximum radius.
    radius_min : `~astropy.units.Quantity`
        Minimum radius.
    nbin : int
        Number of boxes along the line

    Returns
    -------
    regions : list of `~regions.RectangleSkyRegion`
        Regions in which the profiles are made
    """
    regions = []

    edges = np.linspace(radius_min, u.Quantity(radius_max), nbin)

    for r_in, r_out in zip(edges[:-1], edges[1:]):
        region = CircleAnnulusSkyRegion(
            center=center,
            inner_radius=r_in,
            outer_radius=r_out,
        )
        regions.append(region)

    return regions


def region_to_frame(region, frame):
    """Convert a region to a different frame

    Parameters
    ----------
    region : `~regions.SkyRegion`
        region to transform
    frame : "icrs" or "galactic"
        frame to tranform the region into

    Returns
    -------
    region_new : `~regions.SkyRegion`
        region in the given frame
    """
    from gammapy.maps import WcsGeom

    wcs = WcsGeom.create(skydir=region.center, binsz=0.01, frame=frame).wcs
    region_new = region.to_pixel(wcs).to_sky(wcs)
    return region_new


def region_circle_to_ellipse(region):
    """Convert a CircleSkyRegion to an EllipseSkyRegion

    Parameters
    ----------
    region : `~regions.CircleSkyRegion`
        region to transform

    Returns
    -------
    region_new : `~regions.EllipseSkyRegion`
        Elliptical region with same major and minor axis
    """

    region_new = EllipseSkyRegion(
        center=region.center, width=region.radius, height=region.radius
    )
    return region_new


def containment_region(integral_map, fraction=0.68, n_levels=100, apply_union=True):
    """Find the iso-contours region corresponding to a given containment
        for a map of integral quantities.

    Parameters
    ----------
    integral_map : `~gammapy.maps.WcsNDMap`
        Map of integral quantities
    fraction : float
        Containment fraction
    n_levels : int
        Numbers of contours levels used to find the required containment region.

    Returns
    -------
    regions : list of ~regions.PolygonSkyRegion` or `~regions.CompoundSkyRegion`
        regions from iso-contours matching containment fraction
    """
    integral_map = integral_map.reduce_over_axes()
    fmax = np.nanmax(integral_map.data)
    if fmax != 0.0:
        frange = np.linspace(fmax / n_levels, fmax, n_levels)
        fsum = integral_map.data.sum()
        for fval in frange:
            S = np.sum(integral_map.data[integral_map.data > fval]) / fsum
            if S <= fraction:
                break
        plt.ioff()
        fig = plt.figure()
        cs = plt.contour(integral_map.data.squeeze(), [fval])
        plt.close(fig)
        plt.ion()
        regions_pieces = []
        for kp, pp in enumerate(cs.collections[0].get_paths()):
            vertices = []
            for v in pp.vertices:
                v_coord = integral_map.geom.pix_to_coord(v)
                vertices.append([v_coord[0], v_coord[1]])
            vertices = SkyCoord(vertices, frame=integral_map.geom.frame)
            regions_pieces.append(PolygonSkyRegion(vertices))

        if apply_union:
            # compound from union seems not supported to write in ds9 format
            # so regions_pieces contains a list that is supported
            # while regions_full can be saved as .npz
            regions_union = regions_pieces[0]
            for region in regions_pieces[1:]:
                regions_union = regions_union.union(region)
            return regions_union
        else:
            return regions_pieces


def containment_radius(integral_map, fraction=0.68, n_levels=100):
    """Compute containment radius from the center of a map with integral quantities

    Parameters
    ----------
    integral_map : `~gammapy.maps.WcsNDMap`
        Map of integral quantities
    fraction : float
        Containment fraction
    n_levels : int
        Numbers of contours levels used to find the required containment radius.

    Returns
    -------
    radius : `~regions.CompoundSkyRegion`
        Containement radius
    regions_pieces : list of `~regions.PolygonSkyRegion`


    """
    integral_map = integral_map.reduce_over_axes()
    coords = integral_map.geom.get_coord()
    grid = SkyCoord(coords["lon"], coords["lat"], frame=integral_map.geom.frame)
    center = integral_map.geom.center_skydir
    hwidth = np.max(integral_map.geom.width) / 2.0

    radius = np.nan
    fmax = np.nanmax(integral_map.data)
    if fmax != 0.0:
        rrange = np.linspace(hwidth / n_levels, hwidth, n_levels)
        fsum = integral_map.data.sum()
        for radius in rrange:
            S = np.sum(integral_map.data[grid.separation(center) <= radius]) / fsum
            if S > fraction:
                break
    return radius

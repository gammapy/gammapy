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
    RectangleSkyRegion,
    Regions,
    PolygonSkyRegion,
    PolygonPixelRegion
)

from regions.core.pixcoord import PixCoord
from regions.core.metadata import RegionMeta, RegionVisual
from regions._utils.wcs_helpers import pixel_scale_angle_at_skycoord

__all__ = [
    "compound_region_to_regions",
    "make_concentric_annulus_sky_regions",
    "make_orthogonal_rectangle_sky_regions",
    "regions_to_compound_region",
    "region_to_frame",
]


def compound_region_center(compound_region):
    """Compute center for a CompoundRegion.

    The center of the compound region is defined here as the geometric median
    of the individual centers of the regions. The geometric median is defined
    as the point the minimises the distance to all other points.

    Parameters
    ----------
    compound_region : `CompoundRegion`
        Compound region.

    Returns
    -------
    center : `~astropy.coordinates.SkyCoord`
        Geometric median of the positions of the individual regions.
    """
    regions = compound_region_to_regions(compound_region)

    if len(regions) == 1:
        return regions[0].center

    positions = SkyCoord([region.center.icrs for region in regions])

    def f(x, coords):
        """Function to minimize."""
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
        Compound sky region.

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
        Compound sky region.
    """
    region_union = regions[0]

    for region in regions[1:]:
        region_union = region_union.union(region)

    return region_union

def get_centroid(vertices):
    """Compute centroid of a polygon.
    
    Parameters
    ----------
    vertices : `~astropy.coordinates.SkyCoord`
        List of vertices.

    Returns
    -------
    centroid : `~astropy.coordinates.SkyCoord`
        Centroid of the polygon.
    """
    polygon = []
    for i in range(len(vertices)):
        polygon.append((vertices[i].ra.degree, vertices[i].dec.degree))
    polygon = np.array(polygon)

    # Same polygon, but with vertices cycled around. Now the polygon
    # decomposes into triangles of the form origin-polygon[i]-polygon2[i]
    polygon2 = np.roll(polygon, -1, axis=0)

    # Compute signed area of each triangle
    signed_areas = 0.5 * np.cross(polygon, polygon2)

    # Compute centroid of each triangle
    centroids = (polygon + polygon2) / 3.0

    # Get average of those centroids, weighted by the signed areas.
    centroid = np.average(centroids, axis=0, weights=signed_areas)

    return SkyCoord(centroid[0]*u.deg, centroid[1]*u.deg, frame=vertices.frame)

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

class PolygonPointsSkyRegion(PolygonSkyRegion):
    """Polygon sky region defined by a list of points."""

    def __init__(self, vertices, meta=None, visual=None):
        """Create a polygon sky region.
        
        Parameters
        ----------
        vertices : `~astropy.coordinates.SkyCoord`
            List of vertices.
        meta : `~regions.RegionMeta`, optional
            Region meta data.
        visual : `~regions.RegionVisual`, optional
            Region visual meta data.
        """
        self.vertices = vertices
        self.meta = meta or RegionMeta()
        self.center = get_centroid(vertices)
        self.visual = visual or RegionVisual()

    def to_pixel(self, wcs):
        """Convert to pixel region."""
        x, y = wcs.world_to_pixel(self.vertices)
        center=None
        if self.center is not None:
            center, pixscale, _ = pixel_scale_angle_at_skycoord(self.center, wcs)

        vertices_pix = PixCoord(x, y)
        return PolygonPointsPixelRegion(vertices_pix, center=center, meta=self.meta.copy(),
                                  visual=self.visual.copy())

class PolygonPointsPixelRegion(PolygonPixelRegion):
    """Polygon pixel region defined by a list of points."""

    def __init__(self, vertices, center=None, meta=None, visual=None,
                origin=PixCoord(0, 0)):
        """Create a polygon pixel region.
        
        Parameters
        ----------
        vertices : `~regions.PixCoord`
            List of vertices.
        center : `~regions.PixCoord`, optional
            Center of the region.
        meta : `~regions.RegionMeta`, optional
            Region meta data.
        visual : `~regions.RegionVisual`, optional
            Region visual meta data.
        origin : `~regions.PixCoord`, optional
            Origin of the region.
        """
        self._vertices = vertices
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()
        self.origin = origin
        self.vertices = vertices + origin
        self.center = center

    def to_sky(self, wcs):
        """Convert to sky region.
        
        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            WCS transformation object.

        """
        vertices_sky = wcs.pixel_to_world(self.vertices.x, self.vertices.y)
        #center = None
        #if self.center is not None:
        #    center = wcs.pixel_to_world(self.center.x, self.center.y)

        return PolygonPointsSkyRegion(vertices=vertices_sky, meta=self.meta.copy(), 
                                      visual=self.visual.copy())

    def rotate(self, angle):
        """
        Rotate the region.

        Positive ``angle`` corresponds to counter-clockwise rotation.

        Parameters
        ----------
        center : `~regions.PixCoord`
            The rotation center point.
        angle : `~astropy.coordinates.Angle`
            The rotation angle.

        Returns
        -------
        region : `PolygonPixelRegion`
            The rotated region (which is an independent copy).
        """
        center = self.center - self.origin
        vertices = self.vertices.rotate(center, angle)
        center = self.center.rotate(center, angle)

        return self.copy(vertices=vertices, center=center)

def make_orthogonal_rectangle_sky_regions(start_pos, end_pos, wcs, height, nbin=1):
    """Utility returning an array of regions to make orthogonal projections.

    Parameters
    ----------
    start_pos : `~astropy.regions.SkyCoord`
        First sky coordinate defining the line to which the orthogonal boxes made.
    end_pos : `~astropy.regions.SkyCoord`
        Second sky coordinate defining the line to which the orthogonal boxes made.
    height : `~astropy.quantity.Quantity`
        Height of the rectangle region.
    wcs : `~astropy.wcs.WCS`
        WCS projection object.
    nbin : int, optional
        Number of boxes along the line. Default is 1.

    Returns
    -------
    regions : list of `~regions.RectangleSkyRegion`
        Regions in which the profiles are made.
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
        Center coordinate.
    radius_max : `~astropy.units.Quantity`
        Maximum radius.
    radius_min : `~astropy.units.Quantity`, optional
        Minimum radius. Default is 1e-5 deg.
    nbin : int, optional
        Number of boxes along the line. Default is 11.

    Returns
    -------
    regions : list of `~regions.RectangleSkyRegion`
        Regions in which the profiles are made.
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
    """Convert a region to a different frame.

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region to transform.
    frame : {"icrs", "galactic"}
        Frame to transform the region into.

    Returns
    -------
    region_new : `~regions.SkyRegion`
        Region in the given frame.
    """
    from gammapy.maps import WcsGeom

    wcs = WcsGeom.create(skydir=region.center, binsz=0.01, frame=frame).wcs
    region_new = region.to_pixel(wcs).to_sky(wcs)
    return region_new


def region_circle_to_ellipse(region):
    """Convert a CircleSkyRegion to an EllipseSkyRegion.

    Parameters
    ----------
    region : `~regions.CircleSkyRegion`
        Region to transform.

    Returns
    -------
    region_new : `~regions.EllipseSkyRegion`
        Elliptical region with same major and minor axis.
    """
    region_new = EllipseSkyRegion(
        center=region.center, width=region.radius, height=region.radius
    )
    return region_new

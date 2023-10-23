# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import SkyCoord
from regions import PolygonSkyRegion
import matplotlib.pyplot as plt


def containment_region(map_, fraction=0.68, apply_union=True):
    """Find the iso-contours region corresponding to a given containment
        for a map of integral quantities.

    Parameters
    ----------
    map_ : `~gammapy.maps.WcsNDMap`
        Map of integral quantities
    fraction : float
        Containment fraction
    apply_union : bool
        It True return a compound region otherwise return a list of polygon regions.
        Default is True. Note that compound regions cannot be written in ds9 format but can always be saved with numpy.savez.

    Returns
    -------
    regions : list of ~regions.PolygonSkyRegion` or `~regions.CompoundSkyRegion`
        regions from iso-contours matching containment fraction
    """
    fmax = np.nanmax(map_.data)
    if fmax > 0.0:
        ordered = np.sort(map_.data.flatten())[::-1]
        cumsum = np.nancumsum(ordered)
        ind = np.argmin(np.abs(cumsum / cumsum.max() - fraction))
        fval = ordered[ind]

        plt.ioff()
        fig = plt.figure()
        cs = plt.contour(map_.data.squeeze(), [fval])
        plt.close(fig)
        plt.ion()
        regions_pieces = []
        for pp in cs.collections[0].get_paths():
            vertices = []
            for v in pp.vertices:
                v_coord = map_.geom.pix_to_coord(v)
                vertices.append([v_coord[0], v_coord[1]])
            vertices = SkyCoord(vertices, frame=map_.geom.frame)
            regions_pieces.append(PolygonSkyRegion(vertices))

        if apply_union:
            regions_union = regions_pieces[0]
            for region in regions_pieces[1:]:
                regions_union = regions_union.union(region)
            return regions_union
        else:
            return regions_pieces
    else:
        raise ValueError("No positive values in the map.")


def containment_radius(map_, fraction=0.68, position=None):
    """Compute containment radius from the center of a map with integral quantities

    Parameters
    ----------
    fraction : float
        Containment fraction
    n_levels : int
        Numbers of contours levels used to find the required containment radius.
    position : `~astropy.coordinates.SkyCoord`
        Position from where the containment is conputed.
        Default is the center of the Map.

    Returns
    -------
    radius : `~astropy.coordinates.Angle`
        Minimal radius required to reach the given containement fraction.

    """

    if position is None:
        position = map_.geom.center_skydir

    fmax = np.nanmax(map_.data)
    if fmax > 0.0:
        sep = map_.geom.separation(position).flatten()
        order = np.argsort(sep)
        ordered = map_.data.flatten()[order]
        cumsum = np.nancumsum(ordered)
        ind = np.argmin(np.abs(cumsum / cumsum.max() - fraction))
    else:
        raise ValueError("No positive values in the map.")
    return sep[order][ind]

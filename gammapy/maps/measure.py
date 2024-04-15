# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import SkyCoord
from regions import PolygonSkyRegion
import matplotlib as mpl
import matplotlib.pyplot as plt


def containment_region(map_, fraction=0.393, apply_union=True):
    """Find the iso-contours region corresponding to a given containment
        for a map of integral quantities with a flat geometry.

    Parameters
    ----------
    map_ : `~gammapy.maps.WcsNDMap`
        Map of integral quantities.
    fraction : float, optional
        Containment fraction. Default is 0.393.
    apply_union : bool, optional
        It True return a compound region otherwise return a list of polygon regions.
        Default is True. Note that compound regions cannot be written in ds9 format but can always be saved with numpy.savez.

    Returns
    -------
    regions : list of `~regions.PolygonSkyRegion` or `~regions.CompoundSkyRegion`
        Regions from iso-contours matching containment fraction.
    """
    from . import WcsNDMap

    if not isinstance(map_, WcsNDMap):
        raise TypeError(
            f"This function is only supported for WcsNDMap, got {type(map_)} instead."
        )

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

        try:
            paths_all = cs.get_paths()[0]
            starts = np.where(paths_all.codes == 1)[0]
            stops = np.where(paths_all.codes == 79)[0] + 1
            paths = []
            for start, stop in zip(starts, stops):
                paths.append(
                    mpl.path.Path(
                        paths_all.vertices[start:stop],
                        codes=paths_all.codes[start:stop],
                    )
                )
        except AttributeError:
            paths = cs.collections[0].get_paths()
        for pp in paths:
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


def containment_radius(map_, fraction=0.393, position=None):
    """Compute containment radius from a position in a map
        with integral quantities and flat geometry.

    Parameters
    ----------
    map_ : `~gammapy.maps.WcsNDMap`
        Map of integral quantities.
    fraction : float
        Containment fraction. Default is 0.393.
    position : `~astropy.coordinates.SkyCoord`
        Position from where the containment is computed.
        Default is the center of the Map.

    Returns
    -------
    radius : `~astropy.coordinates.Angle`
        Minimal radius required to reach the given containement fraction.

    """
    from . import WcsNDMap

    if not isinstance(map_, WcsNDMap):
        raise TypeError(
            f"This function is only supported for WcsNDMap, got {type(map_)} instead."
        )

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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..maps import WcsNDMap

__all__ = [
    'make_map_exposure_true_energy',
]


def make_map_exposure_true_energy(pointing, livetime, aeff, geom, offset_max):
    """Compute exposure WcsNDMap in true energy (i.e. not convolved by Edisp).

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area table
    geom : `~gammapy.maps.WcsGeom`
        Reference WcsGeom object used to define geometry (space - energy)
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    expmap : `~gammapy.maps.WcsNDMap`
        Exposure cube (3D) in true energy bins
    """
    offset = geom.separation(pointing)

    # Retrieve energies from WcsNDMap
    # Note this would require a log_center from the geometry
    # Or even better edges, but WcsNDmap does not really allows it.
    energy = geom.axes[0].center * geom.axes[0].unit

    exposure = aeff.data.evaluate(offset=offset, energy=energy)
    exposure *= livetime

    # We check if exposure is a 3D array in case there is a single bin in energy
    # TODO: call np.atleast_3d ?
    if len(exposure.shape) < 3:
        exposure = np.expand_dims(exposure.value, 0) * exposure.unit

    # Put exposure outside offset max to zero
    # This might be more generaly dealt with a mask map
    exposure[:, offset >= offset_max] = 0

    data = exposure.to('m2 s')

    return WcsNDMap(geom, data)

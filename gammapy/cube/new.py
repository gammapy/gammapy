# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.coordinates import Angle
from ..maps import WcsNDMap, Map

__all__ = [
    'make_map_separation',
    'make_map_counts',
    'make_map_exposure_true_energy',
    'make_map_background_irf',
    'make_map_background_fov',
    'fill_map_counts',
]


log = logging.getLogger(__name__)


def make_map_separation(geom, position):
    """Compute distance of pixels to a given position for the input reference WCSGeom.

    Result is returned as a 2D WcsNDmap

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry
    position : `~astropy.coordinates.SkyCoord`
        Reference position

    Returns
    -------
    separation : `~gammapy.maps.Map`
        Separation map (2D)
    """
    # We use WcsGeom.get_coords which does not provide SkyCoords for the moment
    # We convert the output to SkyCoords
    geom = geom.to_image()

    coord = geom.get_coord()
    separation = position.separation(coord.skycoord)

    m = Map.from_geom(geom)
    m.quantity = separation
    return m


def make_map_counts(events, ref_geom, pointing, offset_max):
    """Build a WcsNDMap (space - energy) with events from an EventList.

    The energy of the events is used for the non-spatial axis.

    Parameters
    ----------
    events : `~gammapy.data.EventList`
        Event list
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference WcsGeom object used to define geometry (space - energy)
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    cntmap : `~gammapy.maps.WcsNDMap`
        Count cube (3D) in true energy bins
    """
    counts_map = WcsNDMap(ref_geom)
    fill_map_counts(counts_map, events)

    # Compute and apply FOV offset mask
    offset = make_map_separation(ref_geom, pointing).quantity
    offset_mask = offset >= offset_max
    counts_map.data[:, offset_mask] = 0

    return counts_map


def make_map_exposure_true_energy(pointing, livetime, aeff, ref_geom, offset_max):
    """Compute exposure WcsNDMap in true energy (i.e. not convolved by Edisp).

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area table
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference WcsGeom object used to define geometry (space - energy)
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    expmap : `~gammapy.maps.WcsNDMap`
        Exposure cube (3D) in true energy bins
    """
    offset = make_map_separation(ref_geom, pointing).quantity

    # Retrieve energies from WcsNDMap
    # Note this would require a log_center from the geometry
    # Or even better edges, but WcsNDmap does not really allows it.
    energy = ref_geom.axes[0].center * ref_geom.axes[0].unit

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

    return WcsNDMap(ref_geom, data)


def make_map_background_irf(pointing, livetime, bkg, ref_geom, offset_max):
    """Compute background map from background IRFs.

    TODO: Call a method on bkg that returns integral over energy bin directly
    Related: https://github.com/gammapy/gammapy/pull/1342

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Observation livetime
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset

    Returns
    -------
    background : `~gammapy.maps.WcsNDMap`
        Background predicted counts sky cube in reco energy
    """
    # Compute the expected background
    # TODO: properly transform FOV to sky coordinates
    # For now we assume the background is radially symmetric

    energy_axis = ref_geom.axes[0]
    # Compute offsets of all pixels
    map_coord = ref_geom.get_coord()
    # Retrieve energies from map coordinates
    energy_reco = map_coord[energy_axis.name] * energy_axis.unit
    # TODO: go from SkyCoord to FOV coordinates. Here assume symmetric geometry for fov_lon, fov_lat
    # Compute offset at all the pixels and energy of the Map
    fov_lon = map_coord.skycoord.separation(pointing)
    fov_lat = Angle(np.zeros_like(fov_lon), fov_lon.unit)
    data = bkg.evaluate(fov_lon=fov_lon, fov_lat=fov_lat, energy_reco=energy_reco)

    # TODO: add proper integral over energy
    energy_axis = ref_geom.axes[0]
    d_energy = np.diff(energy_axis.edges) * energy_axis.unit
    d_omega = ref_geom.solid_angle()
    data = (data * d_energy[:, np.newaxis, np.newaxis] * d_omega * livetime).to('').value

    # Put exposure outside offset max to zero
    # This might be more generaly dealt with a mask map
    offset = np.sqrt(fov_lon ** 2 + fov_lat ** 2)
    data[:, offset[0, :, :] >= offset_max] = 0

    return WcsNDMap(ref_geom, data=data)


def make_map_background_fov(acceptance_map, counts_map, exclusion_mask):
    """Build Normalized background map from a given acceptance map and counts map.

    This operation is normally performed on single observation maps.
    An exclusion map is used to avoid using regions with significant gamma-ray emission.
    All maps are assumed to follow the same WcsGeom.

    TODO: A model map could be used instead of an exclusion mask.

    Parameters
    ----------
    acceptance_map : `~gammapy.maps.WcsNDMap`
        Observation hadron acceptance map (i.e. predicted background map)
    counts_map : `~gammapy.maps.WcsNDMap`
        Observation counts map
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask

    Returns
    -------
    norm_bkg_map : `~gammapy.maps.WcsNDMap`
        Normalized background
    """
    # We resize the mask
    mask = np.resize(np.squeeze(exclusion_mask.data), acceptance_map.data.shape)

    # We multiply the data with the mask to obtain normalization factors in each energy bin
    integ_acceptance = np.sum(acceptance_map.data * mask, axis=(1, 2))
    integ_counts = np.sum(counts_map.data * mask, axis=(1, 2))

    # TODO: Here we need to add a function rebin energy axis to have minimal statistics for the normalization

    # Normalize background
    norm_factor = integ_counts / integ_acceptance

    norm_bkg = norm_factor * acceptance_map.data.T

    return acceptance_map.copy(data=norm_bkg.T)

def fill_map_counts(count_map, event_list):
    """Fill events into a counts map.

    The energy of the events is used for a non-spatial axis homogeneous to energy.
    The other non-spatial axis names should have an entry in the column names of the ``EventList``

    Parameters
    ----------
    count_map : `~gammapy.maps.Map`
        Map object, will be filled by this function.
    event_list : `~gammapy.data.EventList`
        Event list
    """
    geom = count_map.geom

    # Make a coordinate dictionary; skycoord is always added
    coord_dict = dict(skycoord=event_list.radec)

    # Now add one coordinate for each extra map axis
    for axis in geom.axes:
        if axis.type == 'energy':
            # This axis is the energy. We treat it differently because axis.name could be e.g. 'energy_reco'
            coord_dict[axis.name] = event_list.energy.to(axis.unit)
        # TODO: add proper extraction for time
        else:
            # We look for other axes name in the table column names (case insensitive)
            colnames = [_.upper() for _ in event_list.table.colnames]
            if axis.name.upper() in colnames:
                column_name = event_list.table.colnames[colnames.index(axis.name.upper())]
                coord_dict.update({axis.name: event_list.table[column_name].to(axis.unit)})
            else:
                raise ValueError("Cannot find MapGeom axis {!r} in EventList".format(axis.name))

    count_map.fill_by_coord(coord_dict)
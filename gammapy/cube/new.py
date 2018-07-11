# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from astropy.nddata import Cutout2D
from astropy.nddata.utils import PartialOverlapError
from ..irf import Background3D
from ..maps import WcsNDMap, WcsGeom, Map
from .basic_cube import fill_map_counts

__all__ = [
    'make_cutout',
    'make_separation_map',
    'make_map_counts',
    'make_map_exposure_true_energy',
    'make_map_hadron_acceptance',
    'make_map_fov_background',
    'MapMaker',
]


def make_cutout(ndmap, position, size, margin='0.1 deg'):
    """Create a cutout of a WcsNDMap around a given direction.

    Parameters
    ----------
    ndmap : `~gammapy.maps.WcsNDMap`
        Map on which the cutout has to be extracted
    position : `~astropy.coordinates.SkyCoord`
        Center position of the cutout box
    size : tuple of `~astropy.coordinates.Angle`
        Angular sizes of the box
    margin : `~astropy.coordinates.Angle`
        Additional safety margin

    Returns
    -------
    cutout : `~gammapy.maps.WcsNDMap`
        The cutout map itself
    cutout_slices : tuple
        Tuple of 1-dim slice objects
    """
    # Here we implicitly assume ndmap has 3 dimensions.
    # We might add a test to check this

    # cutout box size
    size = Angle(size) + Angle(margin)

    # First create a cutout 2D of the ndmap
    cutout2d = Cutout2D(
        data=ndmap.data[0], wcs=ndmap.geom.wcs,
        position=position, size=size, mode='strict'
    )

    # Create the slices with the non-spatial axis
    cutout_slices = tuple([slice(0, ndmap.data.shape[0])]) + cutout2d.slices_original

    # Build the new WcsGeom object
    geom = WcsGeom(cutout2d.wcs, cutout2d.shape[::-1], axes=ndmap.geom.axes)
    data = ndmap.data[cutout_slices]

    ndmap_cutout = WcsNDMap(geom, data)

    return ndmap_cutout, cutout_slices


def make_separation_map(geom, position):
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
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.
    
    Returns
    -------
    cntmap : `~gammapy.maps.WcsNDMap`
        Count cube (3D) in true energy bins
    """
    count_map = WcsNDMap(ref_geom)
    fill_map_counts(count_map, events)

    # Compute and apply FOV offset mask
    offset = make_separation_map(ref_geom, pointing).quantity
    offset_mask = offset >= offset_max
    count_map.data[:, offset_mask] = 0

    return count_map


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
    offset = make_separation_map(ref_geom, pointing).quantity

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


def make_map_exposure_reco_energy(pointing, livetime, aeff, edisp, spectrum, ref_geom, offset_max, etrue_bins):
    """Compute exposure WcsNDMap in reco energy.

    After convolution by Edisp and assuming a true energy spectrum.
    This is useful to perform 2D imaging studies.

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area table
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion table
    spectrum : `~gammapy.spectrum.models`
        Spectral model
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference WcsGeom object used to define geometry (space - energy)
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.
    etrue_bins : `~astropy.units.Quantity`
        True energy bins (edges or centers?)

    Returns
    -------
    expmap : `~gammapy.maps.WcsNDMap`
        Exposure cube (3D) in reco energy bins
    """
    # First Compute exposure in true energy
    # Then compute 4D edisp cube
    # Do the product and sum
    raise NotImplementedError


def make_map_hadron_acceptance(pointing, livetime, bkg, ref_geom, offset_max):
    """Compute hadron acceptance cube i.e.  background predicted counts.

    This function evaluates the background rate model on
    a WcsNDMap, and then multiplies with the cube bin size,
    computed via ???, resulting
    in a cube with values that contain predicted background
    counts per bin. 
    The output cube is - obviously - in reco energy.

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

    # TODO: add a uniform API to the two background classes
    energy_axis = ref_geom.axes[0]
    # Compute offsets of all pixels
    offset = make_separation_map(ref_geom, pointing).quantity
    if isinstance(bkg, Background3D):
        map_coord = ref_geom.get_coord()
        # Compute offsets of all pixels
        detx = offset
        # TODO: go from SkyCoord to FOV coordinates. Here assume symmetric geometry for detx, dety
        dety = Angle(np.zeros_like(detx), detx.unit)
        # Retrieve energies from map coordinates
        energy_reco = map_coord[energy_axis.name] * energy_axis.unit
        data = bkg.evaluate(detx=detx, dety=dety, energy_reco=energy_reco)
    else:
        # Retrieve energies from WcsNDMap
        # Note this would require a log_center from the geometry
        energy = energy_axis.center * energy_axis.unit
        data = bkg.data.evaluate(offset=offset, energy=energy)
        data_shape = ref_geom.shape + offset.shape
        data = np.reshape(data, data_shape)

    # TODO: add proper integral over energy
    energy_axis = ref_geom.axes[0]
    d_energy = np.diff(energy_axis.edges) * energy_axis.unit
    d_omega = ref_geom.solid_angle()
    data = (data * d_energy[:, np.newaxis, np.newaxis] * d_omega * livetime).to('').value

    # Put exposure outside offset max to zero
    # This might be more generaly dealt with a mask map
    data[:, offset >= offset_max] = 0

    return WcsNDMap(ref_geom, data=data)


def make_map_fov_background(acceptance_map, counts_map, exclusion_mask):
    """Build Normalized background map from a given acceptance map and count map.

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
    # TODO: Here we should test that WcsGeom are consistent

    # We resize the mask
    mask = np.resize(np.squeeze(exclusion_mask.data), acceptance_map.data.shape)

    # We multiply the data with the mask to obtain normalization factors in each energy bin
    integ_acceptance = np.sum(acceptance_map.data * mask, axis=(1, 2))
    integ_counts = np.sum(counts_map.data * mask, axis=(1, 2))

    # TODO: Here we need to add a function rebin energy axis to have minimal statistics for the normalization

    # Normalize background
    norm_factor = integ_counts / integ_acceptance

    norm_bkg = norm_factor * acceptance_map.data.T

    return WcsNDMap(acceptance_map.geom, data=norm_bkg.T)


def make_map_ring_background(ring_estimator, acceptance_map, counts_map, exclusion_mask):
    """Estimate background map using rings.

    Build normalized background map from a given acceptance map and count map using
    the ring background technique.
    This operation is performed on single observation maps.
    An exclusion map is used to avoid using regions with significant gamma-ray emission.
    All maps are assumed to follow the same WcsGeom.

    Note that the RingBackgroundEstimator class has to be adapted to support WcsNDMaps.

    Parameters
    ----------
    ring_estimator: `~gammapy.background.AdaptiveRingBackgroundEstimator` or `RingBackgroundEstimator`
        Ring background estimator object
    acceptance_map : `~gammapy.maps.WcsNDMap`
        Hadron acceptance map (i.e. predicted background map)
    counts_map : `~gammapy.maps.WcsNDMap`
        Counts map
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask

    Returns
    -------
    norm_bkg_map : `~gammapy.maps.WcsNDMap`
         the normalized background
    """
    raise NotImplementedError


class MapMaker(object):
    """Make all basic maps for a single observation.

    Parameters
    ----------
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    """

    def __init__(self, ref_geom, offset_max):
        self.offset_max = offset_max
        self.ref_geom = ref_geom

        # We instantiate the end products of the MakeMaps class
        self.count_map = WcsNDMap(self.ref_geom)

        data = np.zeros_like(self.count_map.data)
        self.exposure_map = WcsNDMap(self.ref_geom, data, unit="m2 s")

        data = np.zeros_like(self.count_map.data)
        self.background_map = WcsNDMap(self.ref_geom, data)

        # We will need this general exclusion mask for the analysis
        self.exclusion_map = WcsNDMap(self.ref_geom)
        self.exclusion_map.data += 1

    def process_obs(self, obs):
        """Process one observation.

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            Observation
        """
        # First make cutout of the global image
        try:
            exclusion_mask_cutout, cutout_slices = make_cutout(
                self.exclusion_map, obs.pointing_radec,
                [2 * self.offset_max, 2 * self.offset_max],
            )
        except PartialOverlapError:
            # TODO: can we silently do the right thing here? Discuss
            print("Observation {} not fully contained in target image. Skipping it.".format(obs.obs_id))
            return

        cutout_geom = exclusion_mask_cutout.geom

        count_obs_map = make_map_counts(
            obs.events, cutout_geom, obs.pointing_radec, self.offset_max,
        )

        expo_obs_map = make_map_exposure_true_energy(
            obs.pointing_radec, obs.observation_live_time_duration,
            obs.aeff, cutout_geom, self.offset_max,
        )

        acceptance_obs_map = make_map_hadron_acceptance(
            obs.pointing_radec, obs.observation_live_time_duration,
            obs.bkg, cutout_geom, self.offset_max,
        )

        background_obs_map = make_map_fov_background(
            acceptance_obs_map, count_obs_map, exclusion_mask_cutout,
        )

        self._add_cutouts(cutout_slices, count_obs_map, expo_obs_map, background_obs_map)

    def _add_cutouts(self, cutout_slices, count_obs_map, expo_obs_map, acceptance_obs_map):
        """Add current cutout to global maps."""
        self.count_map.data[cutout_slices] += count_obs_map.data
        self.exposure_map.data[cutout_slices] += expo_obs_map.quantity.to(self.exposure_map.unit).value
        self.background_map.data[cutout_slices] += acceptance_obs_map.data

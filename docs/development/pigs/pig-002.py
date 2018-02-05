import numpy as np

from collections import OrderedDict

from astropy.coordinates import Angle,SkyCoord
import astropy.units as u
from astropy.units import Quantity
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError, PartialOverlapError

from gammapy.maps import WcsNDMap, WcsGeom, MapAxis
from gammapy.data import EventList


def make_cutout(ndmap, position, size, margin = 0.1*u.deg):
    """Create a cutout of a WcsNDMap around a given direction.

    Parameters
    ----------
    ndmap : `~gammapy.maps.WcsNDMap`
            the map on which the cutout has to be extracted
    position : `~astropy.coordinates.SkyCoord`
            the center position of the cutout box
    size : Tuple of `~astropy.coordinates.Angle`
            the angular sizes of the box
    margin : `~astropy.coordinates.Angle`
            additional safety margin 

    Returns
    -------
    cutout : `~gammapy.maps.WcsNDMap`
             the cutout map itself
    cutout_slice : 

    """
    # Here we implicitly assume ndmap has 3 dimensions.
    # We might add a test to check this

    # cutout box size
    size = Quantity(size) + margin
    
    # First create a cutout 2D of the ndmap
    # Use astropy Cutout2D
    
    try:
        cutout2d = Cutout2D(
            data=ndmap.data[0], wcs=ndmap.geom.wcs,
            position=position, size=size, mode='strict'
        )
    except PartialOverlapError:
        print("Observation non fully contained in target map. Abort.")
        raise PartialOverlapError
    
    # Create the slices with the non-spatial axis
    cutout_slices = tuple([slice(0, ndmap.data.shape[0])]) + cutout2d.slices_original

    # Build the new WcsGeom object
    wcs = cutout2d.wcs
    new_geom = WcsGeom(wcs, cutout2d.shape[::-1], axes=ndmap.geom.axes)

    ndmap_cutout = WcsNDMap(new_geom,
                            data=ndmap.data[cutout_slices])

    return ndmap_cutout, cutout_slices


def make_separation_map(ref_geom, position):
    """Compute distance of pixels to a given position for the input reference WCSGeom.
    Result is returned as a 2D WcsNDmap
    
    Parameters
    ----------
    ref_geom : `~gammapy.maps.WcsGeom`
            the reference nd image geometry
    position : `~astropy.coordinates.SkyCoord`
            the position considered
    
    Returns
    -------
    valid_map: `~gammapy.maps.WcsNDMap`
            the separation 2D image
    """

    ## We use WcsGeom.get_coords which does not provide SkyCoords for the moment
    ## We convert the output to SkyCoords
    if ref_geom.coordsys == 'GAL':
        frame = 'galactic'
    elif ref_geom.coordsys =='CEL':
        frame = 'icrs'
    else:
        raise ValueError("Incorrect coordinate system.")

    ## This might break if the WcsNDMap does not have 3D
    tmp_coords = ref_geom.get_coords()
    X = tmp_coords[0]
    Y = tmp_coords[1]
    
    ## The first two coordinates are spatial
    skycoords = SkyCoord(X[0][0],Y[0][0],frame=frame,unit='deg')

    ## Compute separations with reference position
    separations = position.separation(skycoords)

    ## Produce reduced MapGeom (i.e. only spatial axes)
    new_geom = ref_geom.to_image()

    ## Return 2D WcsNDMap containing separations
    return WcsNDMap(new_geom, data = np.squeeze(separations))


def make_map_counts(evts, ref_geom, pointing, offset_max):
    """ Build a WcsNDMap (space - energy) with events from an EventList.
    The energy of the events is used for the non-spatial axis.

    Parameters
    ----------
    evts : `~gammapy.data.EventList`
            the input event list
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference WcsGeom object used to define geometry (space - energy)
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.
    
    Returns
    -------
    cntmap : `~gammapy.maps.WcsNDMap`
        Count cube (3D) in true energy bins
    """

    ## For the moment the treatment of units and celestial systems by MapCoords and MapAxis
    ## does not follow astropy units and skycoord.
    
    myunit = ref_geom.axes[0].unit

    ## Convert events coordinates and energy
    if ref_geom.coordsys == 'GAL':
        tmp = [evts.galactic.l,evts.galactic.b,evts.energy.to(myunit)]
    elif ref_geom.coordsys == 'CEL':
        tmp = [evts.radec.ra,evts.radec.dec,evts.energy.to(myunit)]
    else:
        ## should raise an error here
        raise ValueError("Incorrect coordsys.")

    ## Create map
    cntmap = WcsNDMap(ref_geom)
    ## Fill it
    cntmap.fill_by_coords(tmp)

    ## Compute offsets of all pixels
    offset_map = make_separation_map(ref_geom, pointing)

    ## Put counts outside offset max to zero
    ## This might be more generaly dealt with a mask map
    cntmap.data[:, offset_map.data >= offset_max] = 0
    
    return cntmap


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

    ## Compute offsets of all pixels
    offset_map = make_separation_map(ref_geom, pointing)

    ## Retrieve energies from WcsNDMap
    ## Note this would require a log_center from the geometry
    ## Or even better edges, but WcsNDmap does not really allows it. 
    energy = ref_geom.axes[0].center * ref_geom.axes[0].unit

    ## Compute the exposure
    exposure = aeff.data.evaluate(offset=offset_map.data, energy=energy)
    exposure *= livetime

    ## We check if exposure is a 3D array in case there is a single bin in energy
    if len(exposure.shape)<3:
        exposure = np.expand_dims(exposure,0)
    
    ## Put exposure outside offset max to zero
    ## This might be more generaly dealt with a mask map
    exposure[:, offset_map.data >= offset_max] = 0

    return WcsNDMap(ref_geom, data=exposure)



def make_map_exposure_reco_energy(pointing, livetime, aeff, edisp, spectrum, ref_geom, offset_max):
    """ Compute exposure WcsNDMap in reco energy (i.e. after convolution by Edisp and assuming a true 
    energy spectrum).
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
    
    ## First Compute exposure in true energy
    ## Then compute 4D edisp cube 
    ## Do the product and sum
    raise NotImplementedError


def make_map_hadron_acceptance(pointing, livetime, bkg, ref_geom, offset_max):
    """Compute hadron acceptance cube i.e.  background predicted counts.

    This function evaluates the background rate model on
    a WcsNDMap, and then multiplies with the cube bin size,
    computed via ???, resulting
    in a cube with values that contain predicted background
    counts per bin. 
    The output cube is - obviously - in reco energy.

    Note:
    -----
    bkg.evaluate should be replaced with a function returning directly an integrated bkg flux. 

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Observation livetime
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    ref_cube : `~gammapy.maps.WcsGeom`
        Reference cube used to define geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    background : `~gammapy.maps.WcsNDMap`
        Background predicted counts sky cube in reco energy
    """

    ## Compute offsets of all pixels
    offset_map = make_separation_map(ref_geom, pointing)

    ## Retrieve energies from WcsNDMap
    ## Note this would require a log_center from the geometry
    energy = ref_geom.axes[0].center * ref_geom.axes[0].unit

    ## Compute the expected background
    # TODO: properly transform FOV to sky coordinates
    # For now we assume the background is radially symmetric

    ## For now on we use offset dependent evaluate function
    ## This needs to be changed
    data = bkg.evaluate(offset=offset_map.data, energy=energy)
 
    data *= livetime  ## * ref_cube.bin_size
    ## Comment for now. We need a proper integration method
    #    data = data.to('')

    data_shape = ref_geom.shape + offset_map.data.shape

    data= np.reshape(data,data_shape)
        
    ## Put exposure outside offset max to zero
    ## This might be more generaly dealt with a mask map
    data[:, offset_map.data >= offset_max] = 0

    return WcsNDMap(ref_geom, data=data)

def make_map_FoV_background(acceptance_map, counts_map, excluded_map):
    """ Build Normalized background map from a given acceptance map and count map.
    This operation is normally performed on single observation maps.
    An exclusion map is used to avoid using regions with significant gamma-ray emission.
    All maps are assumed to follow the same WcsGeom.

    Note
    ----
    A model map could be used instead of an exclusion mask.

    Parameters
    ----------
    acceptance_map : `~gammapy.maps.WcsNDMap`
         the observation hadron acceptance map (i.e. predicted background map)
    counts_map : `~gammapy.maps.WcsNDMap`
         the observation counts map 
    excluded_map : `~gammapy.maps.WcsNDMap`
         the exclusion mask 

    Return
    ------
    norm_bkg_map :: `~gammapy.maps.WcsNDMap`
         the normalized background
    """

    ## Here we should test that WcsGeom are consistent

    ## We resize the mask
    mask = np.resize(np.squeeze(excluded_map.data), acceptance_map.data.shape)
    print(mask.sum())
    ## We multiply the data with the mask to obtain normalization factors in each energy bin
    integ_acceptance = np.sum(acceptance_map.data*mask, axis=(1,2))
    integ_counts =  np.sum(counts_map.data*mask, axis=(1,2))
    
    ## Here we need to add a function rebin energy axis to have minimal statistics for the normalization

    ## Normalize background
    norm_factor= integ_counts/integ_acceptance
    print(norm_factor)
    norm_bkg = norm_factor*acceptance_map.data.T
    
    return WcsNDMap(acceptance_map.geom, data = norm_bkg.T)
    

def make_map_ring_background(ring_estimator, acceptance_map, counts_map, excluded_map):
    """ Build normalized background map from a given acceptance map and count map using
    the ring background technique.
    This operation is performed on single observation maps.
    An exclusion map is used to avoid using regions with significant gamma-ray emission.
    All maps are assumed to follow the same WcsGeom.

    Note that the RingBackgroundEstimator class has to be adapted to support WcsNDMaps.

    Parameters
    ----------
    ring_estimator: `~gammapy.background.AdaptiveRingBackgroundEstimator` or `RingBackgroundEstimator`
         the ring background estimator object
    acceptance_map : `~gammapy.maps.WcsNDMap`
         the observation hadron acceptance map (i.e. predicted background map)
    counts_map : `~gammapy.maps.WcsNDMap`
         the observation counts map 
    excluded_map : `~gammapy.maps.WcsNDMap`
         the exclusion mask 

    Return
    ------
    norm_bkg_map :: `~gammapy.maps.WcsNDMap`
         the normalized background
    """
    raise NotImplementedError


## This is just a basic example of what such a class could look like
class MakeMaps(object):
    """ Make all basic maps for a single Observation.

    Parameters
    ----------
    ref_geom : `~gammapy.maps.WcsGeom`
            the reference image geometry
    offset_max : `~astropy.coordinates.Angle`
            maximum offset considered
    """
    def __init__(self, ref_geom, offset_max):
        self.offset_max = offset_max
        self.ref_geom = ref_geom

        ## We instantiate the end products of the MakeMaps class
        self.count_map = WcsNDMap(self.ref_geom)
        self.exposure_map = WcsNDMap(self.ref_geom)
        self.background_map =  WcsNDMap(self.ref_geom)

        ## We will need this general exclusion mask for the analysis
        self.exclusion_map = WcsNDMap(self.ref_geom)
        self.exclusion_map.data += 1

    def add_exclusion_regions(self,region):
        """ Add exclusion regions to the excluded mask
        """
        raise NotImplementedError
        
        
    def __call__(self,obs, write=None):
        
        # First make cutout of the global image
        try:
            excluded_map_cutout, cutout_slices = make_cutout(self.exclusion_map, obs.pointing_radec ,
                                                             [2*self.offset_max, 2*self.offset_max])
        except PartialOverlapError:
            print("Observation {} not fully contained in target image. Skipping it.".format(obs.obs_id))
            return
        
        cutout_geom = excluded_map_cutout.geom

        # Make count map
        count_obs_map = make_map_counts(obs.events, cutout_geom, obs.pointing_radec, self.offset_max)
        
        # Make exposure map
        expo_obs_map = make_map_exposure_true_energy( obs.pointing_radec, obs.observation_live_time_duration,
                                                      obs.aeff, cutout_geom, self.offset_max)
        
        # Make hadron acceptance map
        acceptance_obs_map = make_map_hadron_acceptance( obs.pointing_radec, obs.observation_live_time_duration,
                                                         obs.bkg, cutout_geom, self.offset_max)
        
        # Make normalized background map
        background_obs_map = make_map_FoV_background(acceptance_obs_map, count_obs_map, excluded_map_cutout)
    
        self._add_cutouts(cutout_slices, count_obs_map, expo_obs_map, background_obs_map)

        

    def _add_cutouts(self,cutout_slices, count_obs_map, expo_obs_map, acceptance_obs_map):
        """ Add current cutout to global maps
        """
        self.count_map.data[cutout_slices] += count_obs_map.data
        self.exposure_map.data[cutout_slices] += expo_obs_map.data
        self.background_map.data[cutout_slices] += acceptance_obs_map.data
        
            
 

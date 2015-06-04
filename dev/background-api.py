###########################################
## API for the gammapy background module ##
###########################################

###### Part 1: toolbox for generating bg models ######

# Use case: read observation table from file
# - given a text file with a list of observations
# - create an observation table
# (optional: read from fits table.)

# High-level pseudo-code:

from gammapy.obs import ObservationTable

obs_table = ObservationTable.read_from_file('obslist.txt')
# Maybe better ... unsure:
obs_table = ObservationTable.read('obslist.txt', format='ascii')
#ref: http://astropy.readthedocs.org/en/latest/api/astropy.table.Table.html#astropy.table.Table.read


# Use case: tool for building an observation table and apply some basic cuts
# - given an observation table (i.e. all observations from a certain observatory)
# - make some selection cuts
# - return filtered observation table
# (similar to H.E.S.S. findruns)

# High-level pseudo-code:

from gammapy.dataset import FindObservations

filtered_obs_table = FindObservations(global_obs_table, cuts...)

# The function should be executable and it should implement cuts in:
# - time (Tmin, Tmax)
# - sky region: circle or box
#   - optional: cut in circle (offset) could have both min and max values for the
#     offset:
#     analysing a large source with the reflected region bg the analyzer may want
#     to avoid runs taken within the ON region, because the analysis would (or
#     should) skip them anyways
# - ...?


# Use case: allow grouping of observations with similar properties
# - given a list of observations (ON or OFF)
# - group them according to similar observation properties (zenith angle,
#   athimut angle, ...?)
# This can later be used to allow the loop over observations to run over
# observation groups instead of over single observations in the bg makers
# (and maybe other places too)

# High-level pseudo-code:

from gammapy.obs import ObservationGroups, ObservationTable

ObservationTable obs_table

class ObservationGroups:
    """Collection of ObservationTable objects grouped according to different
    observation properties."""
# ObservationGroups could also be an ObservationTable or an astropy Table with
# one row per observation and a column OBSERVATION_GROUP_ID = 1, 2, 3, ...
    def __init__(ObservationTable obs_table, array zenith, array azimuth):
    	"""Take a list of observations and group them according to a given
        binning."""
    def get_properties(int obs_table_id):
        """For a certain group of observations, show the common properties
        used for the grouping."""

obs_groups = ObservationGroups(obs_table, zenith=[0, 20, 40, 60], azimuth=[-180, -90, 90, 180])


# Use case: stack observation table
# - given a ObservationTable object, stack the events from each observation into
#   a binned container:
#   -- 1D: offset
#   -- 2D: energy, offset
#   -- 2D: X, Y
#   -- 3D: X, Y, energy

# High-level pseudo-code:

from gammapy.obs import ObservationTable

ObservationTable obs_table

from gammapy.data import EventList

event_list = vstack_from_files(obs_table.get_files())

image = event_list.BinData(X, Y)
#TODO: should I use at this point the existing data containers? (CountsImageDataset, CountsCubeDataset, ...?)

#TODO: what is a good binning for x, y, energy? (and offset?)
#      lin, lin, log? (sqr?)
#      WCS for x, y? (offset?)
#      bin width?
#      boundaries?


# Use case: observation matcher
# - given a list of on observations
# - find matching off observations or matching bg template

# High-level pseudo-code:

from gammapy.obs import BackgroundObservationMatcher

class BackgroundObservationMatcher:
    """Matches observations / observation groups with similar background."""
    def __init__(ObservationTable off_obs):
    def find_best_off_obs(ObservationTable on_obs):
    	"""Called for on / off bg analysis needs to find matching off observations."""
        return ObservationTable
    def find_best_off_template(ObservationTable on_obs):
        """Called for FOV bg analysis, need to find matching off template."""


# Use case: apply exclusion regions
# - given a certain container (image or cube), a similar container with the
# livetime, a certain catalog and a radius
# - apply a mask to exclude the regions around the sources from the catalog
#   and correct the livetime accordingly
# (optional: consider the extensions given in the catalog)

# High-level pseudo-code:

catalog = gammapy.datasets.load_catalog_tevcat()

from background.utils import ApplyExclusionRegions

def ApplyExclusionRegions(image, livetime, catalog, radius, bool consider_cat_ext):
    """Apply a mask to the image (and the livetime image) for the sources listed
    in the catalog. For each source, the mask will be applied for all bins
    around the source position within the specified radius. If consider_cat_ext
    is activated, the mask will be applied to a circular region of radius equals
    the radius from the catalog plus the specified radius.
    TODO: what happens if the source is not circular in the catalog?"""

ApplyExclusionRegions(image, livetime, catalog, radius, False)


# Use case: compute bg models from OFF observations
# - given HESS OFF observation lists
#    (optional: use aldo CTA off observation lists created with ctobssim)
# - define zenith and azimuth bins to group off-observations
# - build a background model for each OFF observation bin
# Model formats:
#   -- 1D: offset
#   -- 2D: energy, offset
#   -- 2D: X, Y
#   -- 3D: X, Y, energy
# Model output: bg IRFs should use the CTA format
# ref: https://github.com/gammapy/gammapy/issues/267

# High-level pseudo-code:
# for 2D (DETX, DETY):
# TODO: DETX, DETY not fully implemented in gammapy.data.EventList.
# TODO: think of other models (1D (offset) or 2D (energy, offset) or 3D
# (X, Y, energy)).

from gammapy.obs import DataStore, ObservationTable, ObservationGroups
from gammapy.data import EventList
from gammapy.background.utils import ApplyExclusionRegions, make_bg_model_set

def make_bg_model_set(obs_groups):
    """Loop over observation groups. For each group:
    - stack obs tables and bin the data.
    - apply exclusion regions."""
    for obs_table in obs_groups:
        event_list = vstack_from_files(obs_table.get_files())
        image = event_list.BinData(X, Y)
        # TODO: get livetime filled image
        catalog = gammapy.datasets.load_catalog_tevcat()
        radius = Quantity(0.3, 'deg')
        ApplyExclusionRegions(image, livetime, catalog, radius, False)

data_store = DataStore('hess')

# Step 1: select observations to use as off observations & group them (zenith, azimuth)
# ref: http://gammapy.readthedocs.org/en/latest/obs/findruns.html
observation_selection = dict(shape='box', frame='galactic',
                             lon=(-120, 70), lat=(-5, 5), border=2)
TODO: deselect observations in Galactic plane
TODO: get a list of good OFF observations as an obs_table
obs_groups = ObservationGroups(obs_table, zenith=[0, 20, 40, 60], azimuth=[-180, -90, 90, 180], )

bg_model_set = make_bg_model_set(obs_groups)

bg_model_set.write(folder_name)


###### Part 2: develop bg modeling methods ######

#TODO: REVIEW!
# Use case: simulate toy bg
# - given a certain set of observation conditions (i.e. zenith/azimuth
# angle, time)
# - simulate bg events according to the CTA background rate FITS files
# - save the events into an event list (optional: save also binned data)
# this function can be used for testing different bg modeling methods:
# does the modeled bg match the background rate used for the simulations?
# reference: https://github.com/gammapy/gammapy/issues/1

# High-level pseudo-code:

telarray = TelArray('cta-prod2')
observation_table = ... # ObservationTable with some distribution of zenith angles
for observation in observation_table:
    event_list = telarray.simulate_background(observation)
    event_list.write_to_folder('sim_data')


#TODO: REVIEW!
# Use case: simulate toy bg
# - given a certain observation time and observation condition (i.e.
#   zenith/azimuth angle)
# - simulate bg events according to a simple analytical model
# - save the events into an event list (optional: save also binned data)
# this function can be used for testing different bg modeling methods:
# does the modeled bg match the analytical model used for the simulations?
# reference: https://github.com/gammapy/gammapy/issues/1

# High-level pseudo-code:

from gammapy.data import EventList
from astropy.units import Quantity
from astropy.coordinates import Angle
from scipy.integrate import quad
from root_numpy import random_sample
from ROOT import TF1

alt = Angle(90 - 30, "deg")
az = Angle(200, "deg")
time = Quantity(20, "h")

##bg_event_list = simulate_toy_bg_one_bin(alt, az, time)

# bg model:
# - spatial coordinates: Gaussian w.r.t. center of FoV
#   TODO: shouldn't this depend on alt, az?
#   desirable: use radial aceptance curves, depending on alt, az.
# - energy: bg spectrum multiplied by the effective area
# for the bg spectrum one can use the Hegra spectrum:
#  power-law with index 2.7
#  ref: http://journals.aps.org/prd/pdf/10.1103/PhysRevD.59.092003
# alternative: use BESS spectrum for low-E cut-off:
# http://iopscience.iop.org/0004-637X/545/2/1135
width = Angle(2.0, "deg")  # for instance, no idea what it should be...
bg_model_spatial = Gaussian(width)
E_0 = Quantity(1, "TeV")
bg_spectrum = NORM_E * pow(E / E_0, 2.7)  # take NORM_E from Hegra paper
# does this exist?
Aeff = find_lookup('effective_area', altitude='alt', azimuth='az')
bg_model_energy = bg_spectrum * Aeff
bg_model = bg_model_spatial * bg_model_energy

# generate bg
n_events = quad(bg_model_energy * Aeff) * time  # integrate in energy

##DETX, DETY = random(bg_model_spatial, n_events)
##energy = random(bg_model_energy, n_events)

offset = random_sample(TF1("f1", bg_model_spatial), n_events, seed=1)
angle = random_sample(TF1("f1", "2*TMath::Pi()"), n_events, seed=1)

DETX, DETY = transform_polar_to_cartesian(offset, angle)

energy = random_sample(TF1("f1", bg_model_energy), n_events, seed=1)

# refs for random generation:
# http://rootpy.github.io/root_numpy/reference/generated/root_numpy.random_sample.html
# http://www.anderswallin.net/2009/05/uniform-random-points-in-a-circle-using-polar-coordinates/
# https://docs.python.org/2/library/random.html

# produce event list
bg_event_list = EventList(DETX, DETY, energy)
# TODO: store header information (alt, az, time)


# Use case: implement FoV bg method for 1 run
# - given the event list of an ON observation
# - model the bg according to the FoV method from Berge 2007 in 2D (X, Y)
#   or 3D (X, Y, E) using an existing bg model
# - deliver absolute numbers (bg statistics) and maps or cubes

# High-level pseudo-code:

from gammapy.background import FovBgMaker

class FovBgMaker():
    """Class for the FoV background maker."""
    def __init__()
    def model(event_list):
        """Model the bg for a given event list."""

        from gammapy.obs import BackgroundObservationMatcher

        # binning: 2D or 3D -> supposing 2D for now
        image_on = event_list.bin_data("DETX", "DETY") ## on map

        # TODO: apply exclusion regions
        image_on_excluded

        bg_template = BackgroundObservationMatcher.search_template(obs_group.properties())
        (optional: calculate template from ON event list, using events outside exclusion regions.)

        # TODO: apply exclusion regions
        bg_template_excluded

        # scale bg template so that counts outside exclusion regions match counts outside exlusion regions in image_on
        bg_template *= image_on_excluded.sum()/bg_template_excluded.sum() ## off map

        image_alpha = 1 ## alpha map (it is one, due to the normalization of the bg template)

        # maps could go into a bgmaps class
        bg_maps_one_obs.onmap = image_on
        ... (same for off and alpha, maybe even livetime/exposure on/off)
        # compute bg stats
        bg_stats.on = bg_maps.onmap.sum() # or restrict to ON region
        ... (similar for off, alpha, ...)

from gammapy.data import EventList

event_list # EventList of a particular observation

from gammapy.background import FovBgMaker

fov_bg_maker = FovBgMaker()

bg_stats, bg_maps = fov_bg_maker.model(event_list)

bg_maps.write(folder_name)
bg_stats.write(folder_name)

# optional: bg is still not subtracted (no excess/significance/TS (map) has been
# calculated), but all ingredients are there, except for user-specific options
# like correlation radius (optional: or smoothing factor)
#           we also need to define an ON region, and map/cube boundaries.

#TODO: I need classes for bg_stats, bg_maps, bg_cubes
#      or maybe group bg_maps/cubes into bg_ndim?


# Use case: implement FoV bg method for a list of observation
# - given a list of ON observations in a text file (optional: fits list?)
# - model the bg according to the FoV method from Berge 2007 in 2D (X, Y)
#   or 3D (X, Y, E) using existing bg models
# - deliver absolute numbers (bg statistics) and maps or cubes

# High-level pseudo-code:

# TODO: discuss
telarray = TelArray('hess')

from gammapy.obs import ObservationTable

# read in obs list file
# see use case "read observation table from file"
obs_table = read file 'obslist.txt'

from obs_id in obs_table.ids: # loop over observations

    from gammapy.data import EventList

    event_list = obs_table.get_obs(obs_id) # EventList of a particular observation

    from gammapy.background import FovBgMaker

    fov_bg_maker = FovBgMaker()

    bg_stats_one_obs, bg_maps_one_obs = fov_bg_maker.model(event_list)

    # stack all maps and stats
    total_bg_maps.stack(bg_maps_one_obs)
    total_bg_stats.stack(bg_stats_one_obs)

total_bg_maps.write(folder_name)
total_bg_stats.write(folder_name)

# optional: bg is still not subtracted


# Use case: implement ring bg method


# Use case: implement ON/OFF bg method
# - given a list of ON observations and a list of OFF observations
# - model the bg from the ON observations according to the ON/OFF method from Berge
#   2007 in 2D (X, Y) or 3D (X, Y, E) developping bg models from the OFF observations
# - deliver absolute numbers (bg statistics) and maps or cubes


# Use case (high-level): GC maps with ON/OFF bg
# - make a significance and flux image of the Galactic center for HESS with
#   on-off background method.
# - Apply on-off background method, i.e. given on- and off-observation list, produce
#   images or cubes for n_on, n_off, exp_on, exp_off.


#TODO: REVIEW!
##########################
## Function definitions ##
##########################

def simulate_toy_bg_one_bin(alt, az, time):
    """
    Function to create dummy bg simulations for test purposes at a particular
    alt-az pair.
    Optional: give number of events, instead of observation time.
    TODO: is there already an established data format?

    Parameters
    ----------
    alt : `~astropy.coordinates.Angle`
        Value of the altitude angle for the simulated observations.
    az: `~astropy.coordinates.Angle`
        Value of the azimuth angle for the simulated observations.
    time : `~astropy.coordinates`
        Value of the simulated observation time.

    Returns
    -------
    image : `~astropy.nddata.NDData`
        Image filled with simulated bg events.

    """


def simulate_toy_bg(time):
    """
    Function to create dummy bg simulations for test purposes.
    Perform several simulations at different alt-az values, calling recursively
    simulate_toy_bg_one_bin.
    Optional: give number of events, instead of observation time.
    TODO: is there already an established data format?

    Parameters
    ----------
    time : `~astropy.coordinates`
        Value of the simulated observation time.

    Returns
    -------
    images : array of `~astropy.nddata.NDData`
        Array of images filled with simulated bg events. One image per alt-az
        pair.

    """


def group_observations(observation_list):
    """
    Function that takes a list of observations and groups them in bins of alt-az. The
    output of the function is one file per alt-az bin with the list of the observations
    of the corresponding bin. It can be used for both, ON and OFF observation_lists.
    TODO: define an alt-az binning!!!
    TODO: how to access observation properties? (i.e. observation header?)!!!

    Parameters
    ----------
    observation_list : `string`
        Name of file with list of observations.

    """


def stack_observations(observation_list):
    """
    Function to stack the events of a observation_list in a 3D cube. It takes a file with
    a observation_list as input and stacks all the events in a `~astropy.nddata.NDData`
    3D cube (x,y, energy).
    Optional: the output could be a 2D histogram (no energy axis) for the ON/OFF
    bg from Berge 2007.
    Optional: the output could be saved into fits files.
    This can be used to stack both the ON or the OFF events of a certain alt-az
    bin as in group_off_observations.
    TODO: think of coordinate system to use!! (nominal system?)!!!
    TODO: mask exclusion regions!!! (and correct obs time/exposure
    accordingly!!!)
    TODO: how to access events? (i.e. events/DST dataset?)!!!

    Parameters
    ----------
    observation_list : `string`
        Name of file with list of observations.

    Returns
    -------
    image : `~astropy.nddata.NDData`
        Image filled with stacked events.

    """


def subtract_onoff_bg_image(image_on, image_off):
    """
    Function to subtract the background events from the ON image using the OFF
    image events using the ON/OFF method.
    Optional: the output could be saved into fits files.
    This can be used to subtract the background of a certain alt-az bin as in
    group_off_observations.
    TODO: need to calculate exposure ratio between ON and OFF (alpha)!!!

    Parameters
    ----------
    image_on : `~astropy.nddata.NDData`
        Histogram with ON events.
    image_off : `~astropy.nddata.NDData`
        Histogram with OFF events.

    Returns
    -------
    image_excess : `~astropy.nddata.NDData`
        Image with background subtracted.
    """


def subtract_onoff_bg_observation_list(observation_list_on, observation_list_off):
    """
    Function to subtract the background from the ON observation_list using the OFF
    observation_list using the ON/OFF method.
    Optional: check if appropriate bg models exist on file.
    Optional: the output could be saved into fits files.
    1) Call group_observations for observation_list_on and observation_list_off, to split the observation_lists
    into alt-az bins.
    2) Call stack_observations for each alt-az bin for ON and OFF.
    3) Call subtract_onoff_bg_image for each alt-az bin.

    Parameters
    ----------
    observation_list_on : `string`
        Name of file with list of ON observations.
    observation_list_off : `string`
        Name of file with list of OFF observations.

    Returns
    -------
    image_excess : `~astropy.nddata.NDData`
        Image with background subtracted.
    """

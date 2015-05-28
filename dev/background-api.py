# API for the gammapy background module

# Use case: perform bg models from OFF runs
# - given HESS off run event lists (one event list per run) (or CTA off run
#   event lists created with ctobssim)
# - select OFF runs and define zenith and azimuth bins to group off-runs
# - build a background model (could be 1D (offset) or 2D (energy, offset) or
#   2D (X, Y) or 3D (X, Y, energy)) for each off-run bin

# High-level pseudo-code (could actually be code that runs a month from now)
# for 2D (DETX, DETY):
# TODO: DETX, DETY not fully implemented in gammapy.data.EventList.
# TODO: think of other models (1D (offset) or 2D (energy, offset) or 3D
# (X, Y, energy)).

from gammapy.obs import DataStore
from gammapy.obs import ObservationTable, RunList
from gammapy.obs import RunGroups
from gammapy.background import make_1d_background_model

data_store = DataStore(‘hess’)

# Step 1: select runs to use as off runs & group them (zenith, azimuth
observation_selection = dict(shape='box', frame='galactic',
                             lon=(-120, 70), lat=(-5, 5), border=2)
run_groups = RunGroups(zenith=[0, 20, 40, 60], deselect runs in Galactic plane...)
# run_groups could be an astropy Table with one row per run
# and a column RUN_GROUP_ID = 1, 2, 3, ...

# TODO: what about exclusion regions???
-> gammapy.datasets.load_catalog_tevcat() and cut out circle with radius 0.3 deg.
-> correct livetime accordingly

parameters = dict(smooth=0.1 deg, …, cutout_agn=True)
bg_model_set = make_bg_model_set(run_groups, parameters)
# start of implementation of the make_bg_model_set function
from group_id in run_groups.ids:
    event_list_filenames = run_groups.get_event_list_filename(group_id)
    # offset_bg_hist is an object of class gammapy.data.CountsOffsetHistogram
counts_xy_hist = histogram_xy(run_lists)
livetime_xy_hist =

bg_model_set.write(folder_name)


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


# Use case: implement FoV bg method
# - given a list of ON runs
# - model the bg according to the FoV method from Berge 2007 in 2D (X, Y)
#   or 3D (X, Y, E) using existing bg models
# - deliver absolute numbers (bg statistics) and maps or cubes

# High-level pseudo-code:

# TODO: discuss
telarray = TelArray('hess')

from gammapy.obs import ObservationTable
obs_table = ObservationTable.read_from_file(‘obslist.txt’)
# Maybe better … unsure:
obs_table = ObservationTable.read(‘obslist.txt’, format=’ascii’)
#ref: http://astropy.readthedocs.org/en/latest/api/astropy.table.Table.html#astropy.table.Table.read

from gammapy.obs import ObservationGroups, ObservationTable
from gammapy.obs import BackgroundObservationMatcher

class BackgroundObservationMatcher:
    “””“Matches runs / run groups with similar background.”””
    def __init__(off_obs)
    def find_best_off_obs(ObservationTable on_obs, int n_obs)
          “””Called for on / off bg analysis needs to find matching off observations.”””
          return ObservationTable
    def find_best_off_obsgroup(on_obs : ObservationTable):
          “””Called for FOV bg analysis, need to find matching off template.”””

obs_groups = ObservationGroups(obs_table, zenith=[0, 20, 40, 60], azimuth=[0, 180, 360])

from group_id in obs_groups.ids:
    obs_group = obs_groups.get_obs_group(group_id) # this is an ObservationTable
    event_list = obs_group.stack_obs
    # TODO: apply exclusion regions + correct livetime accordingly
    # search bg template (2D or 3D) for this group
    from gammapy.background import FovBgMaker
    fov_bg_maker = FovBgMaker()
    bg_template = fov_bg_maker.search_template(obs_group.properties()) ## off map
    # apply template in order to subtract bg: to the event list or to binned
    # data (i.e. image or cube)? -> let's bin the data for now, supposing 2D
    image_on = event_list.bin_data("DETX", "DETY") ## on map
    image_alpha = livetime_off / livetime_on ## alpha map
    # maps could go into a bgmaps class
    bg_maps_one_obs.onmap = image_on
    ... (same for off and alpha, maybe even livetime/exposure on/off)
    # compute bg stats
    bg_stats_one_obs.on = bg_maps_one_obs.onmap.sum() # or restrict to ON region
    ... (similar for off, alpha, ...)
    # stack all maps and stats
    total_bg_maps.stack(bg_maps_one_obs)
    total_bg_stats.stack(bg_stats_one_obs)

# optional: bg is still not subtracted (no excess/significance/TS (map) has been
# calculated), but all ingredients are there, except for user-specific options
# like correlation radius (optional: or smoothing factor)
#           we also need to define an ON region, and map/cube boundaries.

total_bg_maps.write(folder_name)
total_bg_stats.write(folder_name)


# Use case: implement ring bg method


# Use case: implement ON/OFF bg method
# - given a list of ON runs and a list of OFF runs
# - model the bg from the ON runs according to the ON/OFF method from Berge
#   2007 in 2D (X, Y) or 3D (X, Y, E) developping bg models from the OFF runs
# - deliver absolute numbers (bg statistics) and maps or cubes


# Use case (high-level): GC maps with ON/OFF bg
# - make a significance and flux image of the Galactic center for HESS with
#   on-off background method.
# - Apply on-off background method, i.e. given on- and off-run list, produce
#   images or cubes for n_on, n_off, exp_on, exp_off.


# Use case: compute background IRFs in CTA format from HESS off runs
# ref: https://github.com/gammapy/gammapy/issues/267


# Function definitions

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


def group_runs(runlist):
    """
    Function that takes a list of runs and groups them in bins of alt-az. The
    output of the function is one file per alt-az bin with the list of the runs
    of the corresponding bin. It can be used for both, ON and OFF runlists.
    TODO: define an alt-az binning!!!
    TODO: how to access run properties? (i.e. run header?)!!!

    Parameters
    ----------
    runlist : `string`
        Name of file with list of runs.

    """


def stack_runs(runlist):
    """
    Function to stack the events of a runlist in a 3D cube. It takes a file with
    a runlist as input and stacks all the events in a `~astropy.nddata.NDData`
    3D cube (x,y, energy).
    Optional: the output could be a 2D histogram (no energy axis) for the ON/OFF
    bg from Berge 2007.
    Optional: the output could be saved into fits files.
    This can be used to stack both the ON or the OFF events of a certain alt-az
    bin as in group_off_runs.
    TODO: think of coordinate system to use!! (nominal system?)!!!
    TODO: mask exclusion regions!!! (and correct obs time/exposure
    accordingly!!!)
    TODO: how to access events? (i.e. events/DST dataset?)!!!

    Parameters
    ----------
    runlist : `string`
        Name of file with list of runs.

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
    group_off_runs.
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


def subtract_onoff_bg_runlist(runlist_on, runlist_off):
    """
    Function to subtract the background from the ON runlist using the OFF
    runlist using the ON/OFF method.
    Optional: check if appropriate bg models exist on file.
    Optional: the output could be saved into fits files.
    1) Call group_runs for runlist_on and runlist_off, to split the runlists
    into alt-az bins.
    2) Call stack_runs for each alt-az bin for ON and OFF.
    3) Call subtract_onoff_bg_image for each alt-az bin.

    Parameters
    ----------
    runlist_on : `string`
        Name of file with list of ON runs.
    runlist_off : `string`
        Name of file with list of OFF runs.

    Returns
    -------
    image_excess : `~astropy.nddata.NDData`
        Image with background subtracted.
    """

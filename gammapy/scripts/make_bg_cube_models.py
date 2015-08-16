# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import logging
log = logging.getLogger(__name__)
import numpy as np
from astropy.coordinates import Angle, SkyCoord

from ..utils.scripts import get_parser, set_up_logging_from_args
from ..obs import (ObservationTable, DataStore, ObservationGroups,
                   ObservationGroupAxis)
from ..datasets import load_catalog_tevcat
from ..data import EventListDataset
from ..background import make_bg_cube_model
# TODO: revise imports!!!

__all__ = ['make_bg_cube_models',
           'create_bg_observation_list',
           'group_observations',
           'stack_observations',
           ]


DEBUG = 1 # 0: no output, 1: output, 2: NOTHING, 3: more verbose
# TODO: remove the DEBUG global variable, when the logger works!!!

def main(args=None):
    parser = get_parser(make_bg_cube_models)
    parser.add_argument('fitspath', type=str,
                        help='Dir path to input event list fits files.')
    parser.add_argument('--test', type=bool, default=False,
                        help='If true, use a subset of observations '
                        'for testing purposes')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)

    set_up_logging_from_args(args)

    make_bg_cube_models(**vars(args))


def make_bg_cube_models(fitspath,
                        test):
    """Create background cube models from the complete dataset of an experiment.

    Starting with gamma-ray event lists and effective area IRFs,
    make background templates. Steps

    1. make a global event list from a datastore
    2. filter the runs keeping only the ones far from known sources
    3. group the runs according to similar observation conditions (i.e. alt, az)
        * using `~gammapy.obs.ObservationGroups`
    4. create a bg cube model for each group using:
        * the `~gammapy.background.make_bg_cube_model` method
        * and `~gammapy.background.CubeBackgroundModel` objects as containers

    The models are stored into FITS files.

    It can take a few minutes to run. For a quicker test, please activate the
    **test** flag.

    TODO: revise doc!!!

    Parameters
    ----------
    fitspath : str
        Path to dir containing event list fits files and a list of them.
    """
    create_bg_observation_list(fitspath, test)
    group_observations(test)
    stack_observations(fitspath)


def create_bg_observation_list(fits_path, test):
    """Make total observation list and filter the observations.

    In a first version, all obs taken within 3 deg of a known source
    will be rejected. If a source is extended, twice the extension is
    added to the corresponding exclusion region radius of 3 deg.

    TODO: on a second version, one could only filter out the runs
          too close to the galacic plane, and afterwards use masks
          to cut out sources, for runs taken on extragalactic objects
          (correcting the livetime accordingly).

    TODO: move function to background/obs module? But where?!!!

    Parameters
    ----------
    fits_path : str
        path to dir containing list of input fits event files
    """
    if DEBUG:
        print()
        print("#######################################")
        print("# Starting create_bg_observation_list #")
        print("#######################################")

    # get full list of H.E.S.S. observations
    data_store = DataStore(dir=fits_path, scheme='hess')
    observation_table = data_store.make_observation_table()

    # for testing, only process a small subset of observations
    if test:
        observation_table = observation_table.select_linspace_subset(num=100)
    if DEBUG:
        print()
        print("full observation table")
        print(observation_table)

    # filter observations: load catalog and reject obs too close to sources

    # load catalog: TeVCAT (no H.E.S.S. catalog)
    catalog = load_catalog_tevcat()

    # for testing, only process a small subset of sources
    if test:
        catalog = catalog[:10]
    if DEBUG:
        print()
        print("TeVCAT catalogue")
        print(catalog)
        print("colnames: {}".format(catalog.colnames))

    # sources coordinates
    sources_coord = SkyCoord(catalog['coord_ra'], catalog['coord_dec'])

    # sources sizes (x, y): radius
    sources_size = Angle([catalog['size_x'], catalog['size_y']])
    sources_size = sources_size.reshape(len(catalog), 2)
    # substitute nan with 0
    sources_size[np.isnan(sources_size)] = 0
    # sources max size
    sources_max_size = np.amax(sources_size, axis=1)

    # sources exclusion radius = 2x max size + 3 deg (fov + 0.5 deg?)
    sources_excl_radius = 2*sources_max_size + Angle(3., 'degree')

    # mask all obs taken within the excl radius of any of the sources
    # loop over sources
    obs_coords = SkyCoord(observation_table['RA'], observation_table['DEC'])
    for i_source in range(len(catalog)):
        selection = dict(type='sky_circle', frame='icrs',
                         lon=sources_coord[i_source].ra,
                         lat=sources_coord[i_source].dec,
                         radius=sources_excl_radius[i_source],
                         inverted = True,
                         border=Angle(0., 'degree'))
        observation_table = observation_table.select_observations(selection)

    # TODO: is there a way to quickly filter out sources in a region of the sky, where H.E.S.S. can't observe????!!!! -> don't loose too much time on this (detail)

    # save the bg observation list to a fits file
    outdir = os.environ['PWD'] + '/'
    outfile = outdir + 'bg_observation_table.fits.gz'
    if DEBUG:
        print("outfile", outfile)
    observation_table.write(outfile, overwrite=True)


def group_observations(test):
    """Group list of observations runs according to observation properties.

    The observations are grouped into observation groups (bins) according
    to their altitude and azimuth angle.
    """
    if DEBUG:
        print()
        print("###############################")
        print("# Starting group_observations #")
        print("###############################")

    # read bg observation table from file
    indir = os.environ['PWD'] + '/'
    infile = indir + 'bg_observation_table.fits.gz'
    observation_table = ObservationTable.read(infile)

    # define observation binning
    altitude_edges = Angle([0, 20, 23, 27, 30, 33, 37, 40, 44, 49, 53, 58, 64, 72, 90], 'degree')
    azimuth_edges = Angle([-90, 90, 270], 'degree')

    # for testing, only process a small subset of bins
    if test:
        altitude_edges = Angle([0, 45, 90], 'degree')
        azimuth_edges = Angle([90, 270], 'degree')

    # define axis for the grouping
    list_obs_group_axis = [ObservationGroupAxis('ALT', altitude_edges, 'bin_edges'),
                           ObservationGroupAxis('AZ', azimuth_edges, 'bin_edges')]

    # create observation groups
    observation_groups = ObservationGroups(list_obs_group_axis)
    if DEBUG:
        print()
        print("observation group axes")
        print(observation_groups.info)
        print("observation groups table (group definitions)")
        print(observation_groups.obs_groups_table)

    # group observations in the obs table according to the obs groups
    observation_table_grouped = observation_table

    # wrap azimuth angles to [-90, 270) deg because of the definition
    # of the obs group azimuth axis
    azimuth = Angle(observation_table_grouped['AZ']).wrap_at(Angle(270., 'degree'))
    observation_table_grouped['AZ'] = azimuth

    # apply grouping
    observation_table_grouped = observation_groups.group_observation_table(observation_table_grouped)

    # wrap azimuth angles back to [0, 360) deg
    azimuth = Angle(observation_table_grouped['AZ']).wrap_at(Angle(360., 'degree'))
    observation_table_grouped['AZ'] = azimuth

    if DEBUG:
        print()
        print("observation table grouped")
        print(observation_table_grouped)

    # save the observation groups and the grouped bg observation list to file
    outdir = os.environ['PWD'] + '/'
    outfile = outdir + 'bg_observation_groups.ecsv'
    if DEBUG:
        print("outfile", outfile)
    observation_groups.write(outfile, overwrite=True)
    outfile = outdir + 'bg_observation_table_grouped.fits.gz'
    if DEBUG:
        print("outfile", outfile)
    observation_table_grouped.write(outfile, overwrite=True)


def stack_observations(fits_path):
    """Stack events for each observation group (bin) and make background model.

    The models are stored into FITS files.

    Parameters
    ----------
    fits_path : str
        path to dir containing list of input fits event files
    """
    if DEBUG:
        print()
        print("###############################")
        print("# Starting stack_observations #")
        print("###############################")

    # read observation grouping and grouped observation table
    indir = os.environ['PWD'] + '/'
    infile = indir + 'bg_observation_groups.ecsv'
    observation_groups = ObservationGroups.read('bg_observation_groups.ecsv')
    infile = indir + 'bg_observation_table_grouped.fits.gz'
    observation_table_grouped = ObservationTable.read(infile)

    # create output folder if not existing
    outdir = os.environ['PWD'] + '/bg_cube_models/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    else:
        # clean folder if available
        for oldfile in os.listdir(outdir):
            os.remove(outdir + oldfile)

    # loop over observation groups
    groups = observation_groups.list_of_groups
    if DEBUG:
        print()
        print("list of groups", groups)

    for group in groups:
        if DEBUG:
            print()
            print("group", group)

        # get group of observations
        observation_table = observation_groups.get_group_of_observations(observation_table_grouped, group)
        if DEBUG:
            print(observation_table)

        # skip bins with no observations
        if len(observation_table) == 0:
            print("WARNING, group {} is empty.".format(group))
            continue # skip the rest

        # create bg cube model
        events_cube, livetime_cube, bg_cube = make_bg_cube_model(observation_table, fits_path, DEBUG)

        # save model to file
        bg_cube_model = bg_cube
        # TODO: store also events_cube, livetime_cube !!!
        outfile = outdir +\
                 'bg_cube_model_group{}'.format(group)
        if DEBUG:
            print("outfile", '{}_table.fits.gz'.format(outfile))
            print("outfile", '{}_image.fits.gz'.format(outfile))
        bg_cube_model.write('{}_table.fits.gz'.format(outfile), format='table')
        bg_cube_model.write('{}_image.fits.gz'.format(outfile), format='image')

        # TODO: bg cube file names won't match the names from michael mayer!!! (also the observation lists: split/unsplit)
        #       the current naming makes it difficult to compare 2 sets of cubes!!!
        # TODO: support 2 namings: groupX, or axis1X_axis2Y_etc !!!
        #       this is still not perfect, since the same var with  different binning produces the same indexing, but it's at least something (and helps comparing to Michi, if the same binning is used)
        # add flag split obs list in observation_groups.group_observation_table??!!!

    # TODO: use random data (write a random data generator (see bg API))
    #       what about IRFs (i.e. Aeff for the E_THRESH?)?

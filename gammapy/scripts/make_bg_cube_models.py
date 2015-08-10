# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import logging
import numpy as np
from astropy.coordinates import Angle, SkyCoord

from ..utils.scripts import get_parser
from ..obs import ObservationTable, DataStore
from .. import datasets
from ..data import EventListDataset
from ..background import make_bg_cube_model
# TODO: revise imports!!!

__all__ = ['make_bg_cube_models',
           'create_bg_observation_list',
           'group_observations',
           'stack_observations',
           ]


DEBUG = 1 # 0: no output, 1: output, 2: run fast, 3: more verbose
# TODO: remove the DEBUG global variable, when the logger works!!!

def main(args=None):
    parser = get_parser(make_bg_cube_models)
    parser.add_argument("-l", "--log", dest="loglevel",
                        choices=['debug', 'info', 'warning', 'error',
                                 'critical'],
                        help="Set the logging level")
    parser.add_argument('fitspath', type=str,
                        help='Dir path to input event list fits files.')
    args = parser.parse_args(args)
    make_bg_cube_models(**vars(args))


def make_bg_cube_models(loglevel,
                        fitspath):
    """Create background cube models from the complete dataset of an experiment.

    Starting with gamma-ray event lists and effective area IRFs,
    make background templates. Steps

    1. make a global event list from a datastore
    2. filter the runs keeping only the ones far from known sources
    3. group the runs according to similar observation conditions (i.e. alt, az)
    4. create a bg cube model for each group using:
        * the `~gammapy.background.make_bg_cube_model` method
        * and `~gammapy.background.CubeBackgroundModel` objects as containers

    The models are stored into FITS files.

    It can take about 10 minutes to run.

    TODO: revise doc!!!

    Parameters
    ----------
    loglevel : str
        Level for the logger.
    fitspath : str
        path to dir containing event list fits files and a list of them
    """
    if (loglevel):
        logging.basicConfig(level=getattr(logging, loglevel.upper()), format='%(levelname)s - %(message)s')

    create_bg_observation_list(fitspath)
    group_observations()
    stack_observations(fitspath)


# define observation binning
# TODO: store it in a file (fits? ascii?) (use an astropy table as help?)!!!

# define a binning in altitude angle
# TODO: ObservationGroups
# https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
altitude_edges = Angle([0, 20, 23, 27, 30, 33, 37, 40, 44, 49, 53, 58, 64, 72, 90], 'degree')
if DEBUG > 1:
    altitude_edges = Angle([0, 45, 90], 'degree')

# define a binning in azimuth angle
# TODO: ObservationGroups
# https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
azimuth_edges = Angle([-90, 90, 270], 'degree')
if DEBUG > 1:
    azimuth_edges = Angle([90, 270], 'degree')


def create_bg_observation_list(fits_path):
    """Make total observation list and filter the observations.

    In a first version, all obs taken within 3 deg of a known source will be rejected. If a source is extended, twice the extension is added to the corresponding exclusion region radius of 3 deg.

    TODO: on a second version, one could only filter out the runs
          too close to the galacic plane, and afterwards use masks
          to cut out sources, for runs taken on extragalactic objects
          (correcting the livetime accordingly).

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
    # TODO: shouldn't observatory='HESS' be a parameter to specify which experiment we are using??!!!
    data_store = DataStore(dir=fits_path)
    observation_table = data_store.make_observation_table()

    # For testing, only process a small subset of observations
    if DEBUG > 1:
        observation_table = observation_table.select_linspace_subset(num=100)
    if DEBUG:
        print()
        print("full observation table")
        print(observation_table)

    # TODO: the column format is not the accepted format in Gammapy!!! -> write converter? or adapt data_store?
    # https://gammapy.readthedocs.org/en/latest/dataformats/observation_lists.html
    # TODO: GLON GLAT are missing the units!!!

    # filter observations: load catalog and reject obs too close to sources

    # load catalog(s): HESS/TeVCAT (what about unpublished sources?)
    # there is no HESS catalog function? (only hess_galactic?)
    catalog = datasets.load_catalog_tevcat()

    # For testing, only process a small subset of sources
    if DEBUG > 1:
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


def group_observations():
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

    # split observation table according to binning

    # define a binning in altitude angle
    # TODO: ObservationGroups
    # https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
    if DEBUG:
        print()
        print("altitude bin boundaries")
        print(repr(altitude_edges))

    # define a binning in azimuth angle
    # TODO: ObservationGroups
    # https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
    if DEBUG:
        print()
        print("azimuth bin boundaries")
        print(repr(azimuth_edges))

    # wrap azimuth angles to (-90, 270) deg
    # TODO: needs re-thinking if azimuth angle definitions change!!!
    #       or if user-defined azimuth angle bins are allowed!!!
    azimuth = Angle(observation_table['AZ_PNT']).wrap_at(Angle(270., 'degree'))
    observation_table['AZ_PNT'] = azimuth

    # create output folder if not existing
    outdir = os.environ['PWD'] + '/splitted_obs_list/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    else:
        # clean folder if available
        for oldfile in os.listdir(outdir):
            os.remove(outdir + oldfile)

    # loop over altitude and azimuth angle bins: remember 1 bin less than bin boundaries
    for i_alt in range(len(altitude_edges) - 1):
        if DEBUG:
            print()
            print("bin alt", i_alt)
        for i_az in range(len(azimuth_edges) - 1):
            if DEBUG:
                print()
                print("bin az", i_az)

            # filter observation table
            observation_table_filtered = observation_table

            selection = dict(type='par_box', variable='ALT_PNT',
                             value_range=(altitude_edges[i_alt], altitude_edges[i_alt + 1]))
            observation_table_filtered = observation_table_filtered.select_observations(selection)

            selection = dict(type='par_box', variable='AZ_PNT',
                             value_range=(azimuth_edges[i_az], azimuth_edges[i_az + 1]))
            observation_table_filtered = observation_table_filtered.select_observations(selection)

            if DEBUG:
                print(observation_table_filtered)

            # skip bins with no obs
            if len(observation_table_filtered) == 0:
                continue # skip the rest

            # save the observation list to a fits file
            outfile = outdir +\
                     'bg_observation_table_alt{0}_az{1}.fits.gz'.format(i_alt, i_az)
            if DEBUG:
                print("outfile", outfile)
            observation_table_filtered.write(outfile)


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

    # create output folder if not existing
    outdir = os.environ['PWD'] + '/bg_cube_models/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    else:
        # clean folder if available
        for oldfile in os.listdir(outdir):
            os.remove(outdir + oldfile)

    # loop over altitude and azimuth angle bins: remember 1 bin less than bin boundaries
    for i_alt in range(len(altitude_edges) - 1):
        if DEBUG:
            print()
            print("bin alt", i_alt)
        for i_az in range(len(azimuth_edges) - 1):
            if DEBUG:
                print()
                print("bin az", i_az)

            # read group observation table from file
            indir = os.environ['PWD'] + '/splitted_obs_list/'
            infile = indir +\
                     'bg_observation_table_alt{0}_az{1}.fits.gz'.format(i_alt, i_az)
            # skip bins with no obs list file
            if not os.path.isfile(infile):
                print("WARNING, file not found: {}".format(infile))
                continue # skip the rest
            observation_table = ObservationTable.read(infile)

            if DEBUG:
                print(observation_table)

            # create bg cube model
            events_cube, livetime_cube, bg_cube = make_bg_cube_model(observation_table, fits_path, DEBUG)

            # save model to file
            bg_cube_model = bg_cube
            # TODO: store also events_cube, livetime_cube !!!
            outfile = outdir +\
                     'bg_cube_model_alt{0}_az{1}'.format(i_alt, i_az)
            if DEBUG:
                print("outfile", '{}_table.fits.gz'.format(outfile))
                print("outfile", '{}_image.fits.gz'.format(outfile))
            bg_cube_model.write('{}_table.fits.gz'.format(outfile), format='table')
            bg_cube_model.write('{}_image.fits.gz'.format(outfile), format='image')

    # TODO: use random data (write a random data generator (see bg API))
    #       what about IRFs (i.e. Aeff for the E_THRESH?)?

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import shutil
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


def main(args=None):
    parser = get_parser(make_bg_cube_models)
    parser.add_argument('fitspath', type=str,
                        help='Path to dir containing list of input fits event files.')
    parser.add_argument('scheme', type=str,
                        help='Scheme of file naming.')
    parser.add_argument('outdir', type=str,
                        help='Dir path to store the results.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    parser.add_argument('--test', action='store_true',
                        help='If activated, use a subset of '
                        'observations for testing purposes')
    parser.add_argument('--method', type=str, default='default',
                        choices=['default', 'michi'],
                        help='Bg cube model calculation method to apply.'
                        'observations for testing purposes')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)

    set_up_logging_from_args(args)

    make_bg_cube_models(**vars(args))


def make_bg_cube_models(fitspath, scheme, outdir, overwrite, test, method):
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

    Parameters
    ----------
    fitspath : str
        Path to dir containing list of input fits event files.
    scheme : str
        Scheme of file naming.
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recomended for analysis).
    test : bool
        If true, run fast (not recomended for analysis).
    method : {'default', 'michi'}
        Bg cube model calculation method to apply.

    Examples
    --------
    >>> gammapy-make-bg-cube-models -h
    >>> gammapy-make-bg-cube-models /path/to/fits/event_lists/base/dir HESS bg_cube_models
    >>> gammapy-make-bg-cube-models /path/to/fits/event_lists/base/dir HESS bg_cube_models --test
    >>> gammapy-make-bg-cube-models /path/to/fits/event_lists/base/dir HESS bg_cube_models --test --overwrite
    >>> gammapy-make-bg-cube-models /path/to/fits/event_lists/base/dir HESS bg_cube_models --a-la-michi

    """
    # create output folder
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    else:
        if overwrite:
            # delete and create again
            shutil.rmtree(outdir) # recursively
            os.mkdir(outdir)
        else:
            # do not overwrite, hence exit
            s_error = "Cannot continue: directory \'{}\' exists.".format(outdir)
            raise RuntimeError(s_error)

    create_bg_observation_list(fitspath, scheme, outdir, overwrite, test)
    group_observations(outdir, overwrite, test)
    stack_observations(fitspath, outdir, overwrite, method)


def create_bg_observation_list(fits_path, scheme, outdir, overwrite, test):
    """Make total observation list and filter the observations.

    In a first version, all obs taken within 3 deg of a known source
    will be rejected. If a source is extended, twice the extension is
    added to the corresponding exclusion region radius of 3 deg.

    Parameters
    ----------
    fits_path : str
        Path to dir containing list of input fits event files.
    scheme : str
        Scheme of file naming.
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recomended for analysis).
    test : bool
        If true, run fast: skip many runs and catalog sources.
    """
    log.info(' ')
    log.info("#######################################")
    log.info("# Starting create_bg_observation_list #")
    log.info("#######################################")

    # get full list of observations
    data_store = DataStore(dir=fits_path, scheme=scheme)
    observation_table = data_store.make_observation_table()

    # for testing, only process a small subset of observations
    if test and len(observation_table) > 100:
        observation_table = observation_table.select_linspace_subset(num=100)
    log.info(' ')
    log.info("Full observation table:")
    log.info(observation_table)

    # filter observations: load catalog and reject obs too close to sources

    # load catalog: TeVCAT (no H.E.S.S. catalog)
    catalog = load_catalog_tevcat()

    # for testing, only process a small subset of sources
    if test:
        catalog = catalog[:5]

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

    # save the bg observation list to a fits file
    outfile = outdir + '/bg_observation_table.fits.gz'
    log.info("Writing {}".format(outfile))
    observation_table.write(outfile, overwrite=overwrite)


def group_observations(outdir, overwrite, test):
    """Group list of observations runs according to observation properties.

    The observations are grouped into observation groups (bins) according
    to their altitude and azimuth angle.

    Parameters
    ----------
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recomended for analysis).
    test : bool
        If true, run fast: define coarse binning for observation grouping.
    """
    log.info(' ')
    log.info("###############################")
    log.info("# Starting group_observations #")
    log.info("###############################")

    # read bg observation table from file
    indir = outdir
    infile = indir + '/bg_observation_table.fits.gz'
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
    log.info(' ')
    log.info("Observation group axes:")
    log.info(observation_groups.info)
    log.info("Observation groups table (group definitions):")
    log.info(observation_groups.obs_groups_table)

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

    log.info(' ')
    log.info("Observation table grouped:")
    log.info(observation_table_grouped)

    # save the observation groups and the grouped bg observation list to file
    outfile = outdir + '/bg_observation_groups.ecsv'
    log.info("Writing {}".format(outfile))
    observation_groups.write(outfile, overwrite=overwrite)
    outfile = outdir + '/bg_observation_table_grouped.fits.gz'
    log.info("Writing {}".format(outfile))
    observation_table_grouped.write(outfile, overwrite=overwrite)


def stack_observations(fits_path, outdir, overwrite, method='default'):
    """Stack events for each observation group (bin) and make background model.

    The models are stored into FITS files.

    Parameters
    ----------
    fits_path : str
        Path to dir containing list of input fits event files.
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recomended for analysis).
    method : {'default', 'michi'}, optional
        Bg cube model calculation method to apply.
    """
    log.info(' ')
    log.info("###############################")
    log.info("# Starting stack_observations #")
    log.info("###############################")

    # read observation grouping and grouped observation table
    indir = outdir
    infile = indir + '/bg_observation_groups.ecsv'
    observation_groups = ObservationGroups.read(infile)
    infile = indir + '/bg_observation_table_grouped.fits.gz'
    observation_table_grouped = ObservationTable.read(infile)

    # loop over observation groups
    groups = observation_groups.list_of_groups
    log.info(' ')
    log.info("List of groups to process: {}".format(groups))

    for group in groups:
        log.info(' ')
        log.info("Processing group: {}".format(group))

        # get group of observations
        observation_table = observation_groups.get_group_of_observations(observation_table_grouped, group)
        log.info(observation_table)

        # skip bins with no observations
        if len(observation_table) == 0:
            log.warning("Group {} is empty.".format(group))
            continue # skip the rest

        # create bg cube model
        bg_cube_model = make_bg_cube_model(observation_table, fits_path, method)

        # save model to file
        outfile = outdir +\
                 '/bg_cube_model_group{}'.format(group)
        log.info("Writing {}".format('{}_table.fits.gz'.format(outfile)))
        log.info("Writing {}".format('{}_image.fits.gz'.format(outfile)))
        bg_cube_model.write('{}_table.fits.gz'.format(outfile), format='table', clobber=overwrite)
        bg_cube_model.write('{}_image.fits.gz'.format(outfile), format='image', clobber=overwrite)

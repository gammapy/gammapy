# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
TODO: review this code, move what's useful to `background_model.py`.


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from ..extern.pathlib import Path
from ..utils.scripts import get_parser, set_up_logging_from_args
from ..data import (ObservationTable, DataStore, ObservationGroups,
                   ObservationGroupAxis)
# from ..background import make_bg_cube_model

__all__ = ['make_bg_cube_models',
           'create_bg_observation_list',
           'group_observations',
           'stack_observations',
           ]

log = logging.getLogger(__name__)


def make_bg_cube_models_main(args=None):
    parser = get_parser(make_bg_cube_models)
    parser.add_argument('indir', type=str,
                        help='Input directory (that contains the event lists)')
    parser.add_argument('outdir', type=str,
                        help='Dir path to store the results.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    parser.add_argument('--test', action='store_true',
                        help='If activated, use a subset of '
                             'observations for testing purposes')
    parser.add_argument('--method', type=str, default='default',
                        choices=['default', 'michi'],
                        help='Bg cube model calculation method to apply.')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)

    set_up_logging_from_args(args)

    make_bg_cube_models(**vars(args))


def make_bg_cube_models(indir, outdir, overwrite=False, test=False, method='default'):
    """Create background cube models from the complete dataset of an experiment.

    Starting with gamma-ray event lists and effective area IRFs,
    make background templates. Steps

    1. make a global event list from a datastore
    2. filter the runs keeping only the ones far from known sources
    3. group the runs according to similar observation conditions (i.e. alt, az)
        * using `~gammapy.data.ObservationGroups`
    4. create a bg cube model for each group using:
        * the `~gammapy.background.make_bg_cube_model` method
        * and `~gammapy.background.FOVCubeBackgroundModel` objects as containers

    The models are stored into FITS files.

    It can take a few minutes to run. For a quicker test, please activate the
    **test** flag.

    Parameters
    ----------
    indir : str
        Input directory (that contains the event lists)
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recommended for analysis).
    test : bool
        If true, run fast (not recommended for analysis).
    method : {'default', 'michi'}
        Bg cube model calculation method to apply.

    Examples
    --------
    $ gammapy-make-bg-cube-models -h
    $ gammapy-make-bg-cube-models <indir> HESS bg_cube_models
    $ gammapy-make-bg-cube-models <indir> HESS bg_cube_models --test
    $ gammapy-make-bg-cube-models <indir> HESS bg_cube_models --test --overwrite
    $ gammapy-make-bg-cube-models <indir> HESS bg_cube_models --method michi

    """
    Path(outdir).mkdir(exist_ok=overwrite)

    create_bg_observation_list(indir, outdir, overwrite, test)
    group_observations(outdir, overwrite, test)
    stack_observations(indir, outdir, overwrite, method)


def create_bg_observation_list(indir, outdir, overwrite, test):
    """Make total observation list and filter the observations.

    In a first version, all obs taken within 3 deg of a known source
    will be rejected. If a source is extended, twice the extension is
    added to the corresponding exclusion region radius of 3 deg.

    Parameters
    ----------
    indir : str
        Input directory (that contains the event lists)
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recommended for analysis).
    test : bool
        If true, run fast: skip many runs and catalog sources.
    """
    log.info(' ')
    log.info("#######################################")
    log.info("# Starting create_bg_observation_list #")
    log.info("#######################################")

    # get full list of observations
    data_store = DataStore.from_dir(indir)
    observation_table = data_store.obs_table

    # for testing, only process a small subset of observations
    if test and len(observation_table) > 100:
        observation_table = observation_table.select_linspace_subset(num=100)
    log.debug(' ')
    log.debug("Full observation table:")
    log.debug(observation_table)

    # TODO: filter observations: load catalog and reject obs too close to sources

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
    sources_excl_radius = 2 * sources_max_size + Angle(3., 'deg')

    # mask all obs taken within the excl radius of any of the sources
    # loop over sources
    obs_coords = SkyCoord(observation_table['RA'], observation_table['DEC'])
    for i_source in range(len(catalog)):
        selection = dict(type='sky_circle', frame='icrs',
                         lon=sources_coord[i_source].ra,
                         lat=sources_coord[i_source].dec,
                         radius=sources_excl_radius[i_source],
                         inverted=True,
                         border=Angle(0., 'deg'))
        observation_table = observation_table.select_observations(selection)

    # save the bg observation list to a fits file
    outfile = Path(outdir) / 'bg_observation_table.fits.gz'
    log.info("Writing {}".format(outfile))
    observation_table.write(str(outfile), overwrite=overwrite)


def group_observations(outdir, overwrite, test):
    """Group list of observations runs according to observation properties.

    The observations are grouped into observation groups (bins) according
    to their altitude and azimuth angle.

    Parameters
    ----------
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recommended for analysis).
    test : bool
        If true, run fast: define coarse binning for observation grouping.
    """
    log.info(' ')
    log.info("###############################")
    log.info("# Starting group_observations #")
    log.info("###############################")

    # read bg observation table from file
    infile = Path(outdir) / 'bg_observation_table.fits.gz'
    observation_table = ObservationTable.read(str(infile))

    # define observation binning
    altitude_edges = Angle([0, 20, 23, 27, 30, 33, 37, 40, 44, 49, 53, 58, 64, 72, 90], 'deg')
    azimuth_edges = Angle([-90, 90, 270], 'deg')

    # for testing, only process a small subset of bins
    if test:
        altitude_edges = Angle([0, 45, 90], 'deg')
        azimuth_edges = Angle([90, 270], 'deg')

    # define axis for the grouping
    list_obs_group_axis = [ObservationGroupAxis('ALT', altitude_edges, 'bin_edges'),
                           ObservationGroupAxis('AZ', azimuth_edges, 'bin_edges')]

    # create observation groups
    observation_groups = ObservationGroups(list_obs_group_axis)
    log.info(' ')
    log.info("Observation group axes:")
    log.info(observation_groups.info)
    log.debug("Observation groups table (group definitions):")
    log.debug(observation_groups.obs_groups_table)

    # group observations in the obs table according to the obs groups
    observation_table_grouped = observation_table

    # wrap azimuth angles to [-90, 270) deg because of the definition
    # of the obs group azimuth axis
    azimuth = Angle(observation_table_grouped['AZ']).wrap_at(Angle(270., 'deg'))
    observation_table_grouped['AZ'] = azimuth

    # apply grouping
    observation_table_grouped = observation_groups.apply(observation_table_grouped)

    # wrap azimuth angles back to [0, 360) deg
    azimuth = Angle(observation_table_grouped['AZ']).wrap_at(Angle(360., 'deg'))
    observation_table_grouped['AZ'] = azimuth

    log.debug(' ')
    log.debug("Observation table grouped:")
    log.debug(observation_table_grouped)

    # save the observation groups and the grouped bg observation list to file
    outfile = Path(outdir) / 'bg_observation_groups.ecsv'
    log.info("Writing {}".format(outfile))
    observation_groups.write(str(outfile), overwrite=overwrite)

    outfile = Path(outdir) / 'bg_observation_table_grouped.fits.gz'
    log.info("Writing {}".format(outfile))
    observation_table_grouped.write(str(outfile), overwrite=overwrite)


def stack_observations(indir, outdir, overwrite, method='default'):
    """Stack events for each observation group (bin) and make background model.

    The models are stored into FITS files.

    Parameters
    ----------
    indir : str
        Input directory (that contains the event lists)
    outdir : str
        Dir path to store the results.
    overwrite : bool
        If true, run fast (not recommended for analysis).
    method : {'default', 'michi'}, optional
        Bg cube model calculation method to apply.
    """
    log.info(' ')
    log.info("###############################")
    log.info("# Starting stack_observations #")
    log.info("###############################")

    # read observation grouping and grouped observation table
    infile = Path(outdir) / 'bg_observation_groups.ecsv'
    observation_groups = ObservationGroups.read(str(infile))
    infile = Path(outdir) / 'bg_observation_table_grouped.fits.gz'
    observation_table_grouped = ObservationTable.read(str(infile))

    # loop over observation groups
    groups = observation_groups.list_of_groups
    log.info(' ')
    log.info("List of groups to process: {}".format(groups))

    for group in groups:
        log.info(' ')
        log.info("Processing group: {}".format(group))

        # get group of observations
        observation_table = observation_groups.get_group_of_observations(
            observation_table_grouped, group)
        log.debug(observation_table)

        # skip bins with no observations
        if len(observation_table) == 0:
            log.warning("Group {} is empty.".format(group))
            continue  # skip the rest

        # create bg cube model
        bg_cube_model = make_bg_cube_model(observation_table, indir, method)

        # save model to file
        for format in ['table', 'image']:
            filename = 'bg_cube_model_group{}_{}.fits.gz'.format(group, format)
            filename = Path(outdir) / filename
            log.info("Writing {}".format(filename))
            bg_cube_model.write(str(filename), format=format, overwrite=overwrite)

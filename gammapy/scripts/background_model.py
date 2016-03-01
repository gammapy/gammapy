# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import click
import numpy as np
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.table import join as table_join
from ..data import DataStore
from ..data import ObservationTable, ObservationGroupAxis, ObservationGroups
from ..background import CubeBackgroundModel
from ..background import EnergyOffsetBackgroundModel
from ..utils.energy import EnergyBounds
from ..extern.pathlib import Path

click.disable_unicode_literals_warning = True

__all__ = []

log = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


class Config:
    """Directory and file locations"""

    run_list = 'runs.lis'
    clobber = True

    _datastore = None

    @property
    def data_store(self):
        if self._datastore:
            return self._datastore

        ds = DataStore.from_dir(self.datadir)
        self._datastore = ds

        return ds

    @property
    def obs_table(self):
        table = Table.read(self.run_list, format='ascii.csv')
        obs_table = self.data_store.obs_table
        table = table_join(table, obs_table)
        return table

    @property
    def obs_table_grouped_filename(self):
        return str(self.outdir / 'obs.ecsv')

    @property
    def obs_table_grouped(self):
        return ObservationTable.read(self.obs_table_grouped_filename, format='ascii.ecsv')

    @property
    def group_def_filename(self):
        return str(self.outdir / 'group-def.ecsv')


# Global config object ... used to set / get config options from the various sub-commands.
config = Config()


@click.group('background',
             context_settings=CONTEXT_SETTINGS)
@click.argument('datadir', 'Directory where the index files are located.')
@click.argument('outdir', 'Directory where all output files go.')
def background_cli(datadir, outdir):
    """Background models (several steps)
    """
    config.datadir = datadir
    config.outdir = Path(outdir)

    # Make sure the background directory exists
    if not config.outdir.exists():
        config.outdir.mkdir()


@background_cli.command('list')
@click.option('--selection', default='offplane',
              type=click.Choice(['offplane', 'debug', 'old-hap-hd']))
@click.option('--max', 'n_obs_max', default=None, type=int,
              help='Maximum number of observations (default: all)')
def background_list(selection, n_obs_max):
    """Make off run list for background models.

    \b
    * selection='old' is the common subset of:
      - Old run list from HAP background model production
      - Runs available as FITS here
    * selection='new' is all runs where
      - FITS data is available here
      - |GLAT| > 5 (i.e. not in the Galactic plane
      - separation to a TeVCat source > 2 deg

    \b
    Parameters
    ----------
    selection : {'offplane', 'debug', 'old-hap-hd'}
        Observation selection method.
    n_obs_max : int, None
        Maximum number of observations (useful for quick testing)
    """
    if selection == 'offplane':
        obs_table = config.data_store.obs_table[:n_obs_max]
        MIN_GLAT = 5
        mask = np.abs(obs_table['GLAT']) > MIN_GLAT
        obs_table = obs_table[mask]
        obs_table = obs_table[['OBS_ID']]
    elif selection == 'debug':
        obs_table = config.data_store.obs_table[:n_obs_max]
        obs_table = obs_table[['OBS_ID']]
    elif selection == 'old-hap-hd':
        hessroot = os.environ['HESSROOT']
        filename = Path(hessroot) / 'hddst/scripts/lookups/lists/acceptance_runs.csv'
        log.info('Reading {}'.format(filename))
        obs_hess = Table.read(str(filename), format='ascii')
        obs_hess = obs_hess['Run'].data

        filename = config.RUN_LIST_EXPORT
        log.info('Reading {}'.format(filename))
        obs_export = Table.read(str(filename), format='ascii')
        obs_export = obs_export['col1'].data

        obs_common = sorted(set(obs_hess) & set(obs_export))

        log.info('Runs HESS:   {:6d}'.format(len(obs_hess)))
        log.info('Runs export: {:6d}'.format(len(obs_export)))
        log.info('Runs common: {:6d}'.format(len(obs_common)))

        if n_obs_max:
            log.info('Applying max. obs selection: {}'.format(n_obs_max))
            obs_common = obs_common[:n_obs_max]

        table = Table(dict(OBS_ID=obs_common))
    else:
        raise ValueError('Invalid selection: {}'.format(selection))

    log.info('Writing {}'.format(config.run_list))
    obs_table.write(config.run_list, format='ascii.csv')


@background_cli.command('group')
def background_group():
    """Group the background observation list.

    For now we'll just use the zenith binning from HAP.
    """
    obs_table = config.obs_table.copy()

    # Define observation groups
    # zenith_bins = np.array([0, 20, 30, 40, 50, 90])
    zenith_bins = np.array([0, 49, 90])
    # zenith_bins = np.array([0, 30, 90])  # for testing
    axes = [ObservationGroupAxis('ZEN_PNT', zenith_bins, fmt='edges')]
    obs_groups = ObservationGroups(axes)
    log.info(obs_groups.info)

    # Apply observation grouping
    obs_table = obs_groups.apply(obs_table)

    # Store the results
    filename = config.obs_table_grouped_filename
    log.info('Writing {}'.format(filename))
    obs_table.write(str(filename), format='ascii.ecsv')

    filename = config.group_def_filename
    log.info('Writing {}'.format(filename))
    obs_groups.obs_groups_table.write(str(filename), format='ascii.ecsv')


@background_cli.command('model')
@click.option('--modeltype', default='3D',
              type=click.Choice(['3D', '2D']))
def background_model(modeltype):
    """Make background models.

    """

    filename = config.obs_table_grouped_filename
    log.info('Reading {}'.format(filename))
    obs_table = ObservationTable.read(str(filename), format='ascii.ecsv')

    groups = sorted(np.unique(obs_table['GROUP_ID']))
    log.info('Groups: {}'.format(groups))
    for group in groups:
        # Get observations in the group
        idx = np.where(obs_table['GROUP_ID'] == group)[0]
        obs_table_group = obs_table[idx]
        log.info('Processing group {} with {} observations'.format(group, len(obs_table_group)))

        # Build the model
        if modeltype == "3D":
            model = CubeBackgroundModel.define_cube_binning(obs_table_group, method='default')
            model.fill_obs(obs_table_group, config.data_store)
            model.smooth()
            model.compute_rate()

            # Store the model
            filename = config.outdir / 'background_{}_group_{:03d}_table.fits.gz'.format(modeltype, group)
            log.info('Writing {}'.format(filename))
            model.write(str(filename), format='table', clobber=config.clobber)

            filename = config.outdir / 'background_{}_group_{:03d}_image.fits.gz'.format(modeltype, group)
            log.info('Writing {}'.format(filename))
            model.write(str(filename), format='image', clobber=config.clobber)

        elif modeltype == "2D":
            ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
            offset = Angle(np.linspace(0, 2.5, 100), "deg")
            model = EnergyOffsetBackgroundModel(ebounds, offset)
            model.fill_obs(obs_table_group, config.data_store)
            model.compute_rate()

            # Store the model
            filename = config.outdir / 'background_{}_group_{:03d}_table.fits.gz'.format(modeltype, group)
            log.info('Writing {}'.format(filename))
            model.write(str(filename))

        else:
            raise ValueError("Invalid model type: {}".format(modeltype))

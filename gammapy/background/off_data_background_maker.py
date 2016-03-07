# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import numpy as np
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.table import join as table_join
from ..data import ObservationTable, ObservationGroupAxis, ObservationGroups
from .models import CubeBackgroundModel
from .models import EnergyOffsetBackgroundModel
from ..utils.energy import EnergyBounds
from ..extern.pathlib import Path

__all__ = [
    'OffDataBackgroundMaker',
]

log = logging.getLogger(__name__)

class OffDataBackgroundMaker(object):
    def __init__(self, data_store, outdir=None, run_list=None, obs_table_grouped_filename=None,
                 group_table_filename=None):
        """
        Parameters
        ----------
        data_store
        run_list
        outdir
        obs_table_grouped_filename
        group_table_filename

        Returns
        -------

        """
        self.data_store = data_store
        if not run_list:
            self.run_list = "run.lis"
        else:
             self.run_list = run_list

        if not outdir:
            self.outdir = "out"
        else:
            self.outdir = outdir

        if not obs_table_grouped_filename:
            self.obs_table_grouped_filename = self.outdir + '/obs.ecsv'
        else:
            self.obs_table_grouped_filename = obs_table_grouped_filename

        if not group_table_filename:
            self.group_table_filename = self.outdir + '/group-def.ecsv'
        else:
            self.group_table_filename = group_table_filename

    def define_obs_table(self):
        table = Table.read(self.run_list, format='ascii.csv')
        obs_table = self.data_store.obs_table
        table = table_join(table, obs_table)
        return table

    def select_observations(self, selection, n_obs_max=None):
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
            obs_table = self.data_store.obs_table[:n_obs_max]
            MIN_GLAT = 5
            mask = np.abs(obs_table['GLAT']) > MIN_GLAT
            obs_table = obs_table[mask]
            obs_table = obs_table[['OBS_ID']]
        elif selection == 'debug':
            obs_table = self.data_store.obs_table[:n_obs_max]
            obs_table = obs_table[['OBS_ID']]
        elif selection == 'old-hap-hd':
            hessroot = os.environ['HESSROOT']
            filename = Path(hessroot) / 'hddst/scripts/lookups/lists/acceptance_runs.csv'
            log.info('Reading {}'.format(filename))
            obs_hess = Table.read(str(filename), format='ascii')
            obs_hess = obs_hess['Run'].data

            filename = self.run_list
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

            obs_table = Table(dict(OBS_ID=obs_common))
        else:
            raise ValueError('Invalid selection: {}'.format(selection))

        log.info('Writing {}'.format(self.run_list))
        obs_table.write(self.run_list, format='ascii.csv')

    def group_observations(self):
        """Group the background observation list.

        For now we'll just use the zenith binning from HAP.
        """

        obs_table = self.define_obs_table()

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
        filename = self.obs_table_grouped_filename
        log.info('Writing {}'.format(filename))
        obs_table.write(str(filename), format='ascii.ecsv')

        filename = self.group_table_filename
        log.info('Writing {}'.format(filename))
        obs_groups.obs_groups_table.write(str(filename), format='ascii.ecsv')

    def make_model(self, modeltype, obs_table=None, excluded_sources=None):
        """Make background models.

        """
        if not obs_table:
            filename = self.obs_table_grouped_filename
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
                model.fill_obs(obs_table_group, self.data_store)
                model.smooth()
                model.compute_rate()

                # Store the model
                filename = self.outdir + '/background_{}_group_{:03d}_table.fits.gz'.format(modeltype, group)
                log.info('Writing {}'.format(filename))
                model.write(str(filename), format='table', clobber = True)

                filename = self.outdir + '/background_{}_group_{:03d}_image.fits.gz'.format(modeltype, group)
                log.info('Writing {}'.format(filename))
                model.write(str(filename), format='image', clobber= True)

            elif modeltype == "2D":
                ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
                offset = Angle(np.linspace(0, 2.5, 100), "deg")
                model = EnergyOffsetBackgroundModel(ebounds, offset)
                model.fill_obs(obs_table_group, self.data_store, excluded_sources)
                model.compute_rate()

                # Store the model
                filename = self.outdir + '/background_{}_group_{:03d}_table.fits.gz'.format(modeltype, group)
                log.info('Writing {}'.format(filename))
                model.write(str(filename), overwrite=True)

            else:
                raise ValueError("Invalid model type: {}".format(modeltype))

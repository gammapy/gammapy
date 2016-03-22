# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.table import join as table_join
from ..data import ObservationTable, ObservationGroupAxis, ObservationGroups
from .models import CubeBackgroundModel
from .models import EnergyOffsetBackgroundModel
from ..utils.energy import EnergyBounds
from ..utils.axis import sqrt_space
from ..extern.pathlib import Path

__all__ = [
    'OffDataBackgroundMaker',
]

log = logging.getLogger(__name__)


class OffDataBackgroundMaker(object):
    """OffDataBackgroundMaker class.

        Class that will select an OFF list run from a Data list and then group this runlist in group of
        zenithal angle and efficiency. Then for each group, it will compute the background rate model in
        3D *(X, Y, energy)* or 2D *(energy, offset)* via the class `~gammapy.background.CubeBackgroundModel` (3D) or
        `~gammapy.background.EnergyOffsetBackgroundModel` (2D).

        Parameters
        ----------
        data_store : `~gammapy.data.DataStore`
            Data for the background model
        run_list : str
            filename where is store the OFF run list
        outdir : str
            directory where will go the output
        obs_table :gamm `~astropy.table.Table`
            observation table of the OFF run List used for the background modelling
            require GROUP_ID column
        ntot_group: int
            Number of group in zenithal angle, efficiency
        excluded_sources : `~astropy.table.Table`
            Table of excluded sources.
            Required columns: RA, DEC, Radius
        """

    def __init__(self, data_store, outdir=None, run_list=None, obs_table=None, ntot_group=None, excluded_sources=None):
        self.data_store = data_store
        if not run_list:
            self.run_list = "run.lis"
        else:
            self.run_list = run_list

        if not outdir:
            self.outdir = "out"
        else:
            self.outdir = outdir
        self.obs_table = obs_table
        self.excluded_sources = excluded_sources

        self.obs_table_grouped_filename = self.outdir + '/obs.ecsv'
        self.group_table_filename = self.outdir + '/group-def.ecsv'

        self.models3D = dict()
        self.models2D = dict()
        self.ntot_group = ntot_group

    def define_obs_table(self):
        """Make an obs table for the OFF runs list.

        This table is created from the obs table of all the runs

        Returns
        -------
        table : `~astropy.table.Table`
            observation table of the OFF run List
        """
        table = Table.read(self.run_list, format='ascii.csv')
        obs_table = self.data_store.obs_table
        table = table_join(table, obs_table)
        return table

    def select_observations(self, selection, n_obs_max=None):
        """Make off run list for background models.

        * selection='offplane' is all runs where
          - Observation is available here
          - abs(GLAT) > 5 (i.e. not in the Galactic plane)
        * selection='all' -- all available observations

        Parameters
        ----------
        selection : {'offplane', 'all'}
            Observation selection method.
        n_obs_max : int, None
            Maximum number of observations (useful for quick testing)
        """
        if selection == 'offplane':
            obs_table = self.data_store.obs_table[:n_obs_max]
            min_glat = 5
            mask = np.abs(obs_table['GLAT']) > min_glat
            obs_table = obs_table[mask]
            obs_table = obs_table[['OBS_ID']]
        elif selection == 'all':
            obs_table = self.data_store.obs_table[:n_obs_max]
            obs_table = obs_table[['OBS_ID']]
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
        axes = [ObservationGroupAxis('ZEN_PNT', [0, 49, 90], fmt='edges')]
        obs_groups = ObservationGroups(axes)
        log.info(obs_groups.info)

        # Apply observation grouping
        obs_table = obs_groups.apply(obs_table)

        # Store the results
        filename = self.obs_table_grouped_filename
        log.info('Writing {}'.format(filename))
        obs_table.write(str(filename), format='ascii.ecsv')
        self.obs_table = obs_table

        filename = self.group_table_filename
        log.info('Writing {}'.format(filename))
        obs_groups.obs_groups_table.write(str(filename), format='ascii.ecsv')
        self.ntot_group = obs_groups.n_groups

    def make_model(self, modeltype):
        """Make background models.

        Create the list of background model (`~gammapy.background.CubeBackgroundModel` (3D) or
        `~gammapy.background.EnergyOffsetBackgroundModel` (2D)) for each group in zenithal angle and efficiency

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        """

        groups = sorted(np.unique(self.obs_table['GROUP_ID']))
        log.info('Groups: {}'.format(groups))
        for group in groups:
            print(group)
            # Get observations in the group
            idx = np.where(self.obs_table['GROUP_ID'] == group)[0]
            obs_table_group = self.obs_table[idx]
            obs_ids = list(obs_table_group['OBS_ID'])
            log.info('Processing group {} with {} observations'.format(group, len(obs_table_group)))

            # Build the model
            if modeltype == "3D":
                model = CubeBackgroundModel.define_cube_binning(obs_table_group, method='default')
                model.fill_obs(obs_table_group, self.data_store)
                model.smooth()
                model.compute_rate()
                self.models3D[str(group)] = model
            elif modeltype == "2D":
                ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 15, 'TeV')
                offset = sqrt_space(start=0, stop=2.5, num=100) * u.deg
                model = EnergyOffsetBackgroundModel(ebounds, offset)
                model.fill_obs(obs_ids=obs_ids, data_store=self.data_store, excluded_sources=self.excluded_sources)
                model.compute_rate()
                self.models2D[str(group)] = model
            else:
                raise ValueError("Invalid model type: {}".format(modeltype))

    def save_model(self, modeltype, ngroup):
        """Save model to fits for one group in zenithal angle and efficiency.

          Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        ngroup : int
            Number of groups
        """
        filename = self.outdir + '/background_{}_group_{:03d}_table.fits.gz'.format(modeltype, ngroup)

        if modeltype == "3D":
            if str(ngroup) in self.models3D.keys():
                self.models3D[str(ngroup)].write(str(filename), format='table', clobber=True)
            else:
                print("No run in the band {}".format(ngroup))
        if modeltype == "2D":
            if str(ngroup) in self.models2D.keys():
                self.models2D[str(ngroup)].write(str(filename), overwrite=True)
            else:
                print("No run in the band {}".format(ngroup))

    def save_models(self, modeltype):
        """Save model to fits for all the groups in zenithal angle and efficiency.

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        """
        for ngroup in range(self.ntot_group):
            self.save_model(modeltype, ngroup)

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
        obs_table : `~astropy.table.Table`
            observation table of the OFF run List used for the background modelling
            require GROUP_ID column
        excluded_sources : `~astropy.table.Table`
            Table of excluded sources.
            Required columns: RA, DEC, Radius
        """
    def __init__(self, data_store, outdir=None, run_list=None, obs_table= None, excluded_sources=None):
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
        self.excluded_sources =excluded_sources

        self.obs_table_grouped_filename = self.outdir + '/obs.ecsv'
        self.group_table_filename = self.outdir + '/group-def.ecsv'

        self.models3D = list()
        self.models2D = list()
        self.ntot_group = None

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
            min_glat = 5
            mask = np.abs(obs_table['GLAT']) > min_glat
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
        modeltype : str
            type of the background modelisation: 3D or 2D

        """

        groups = sorted(np.unique(self.obs_table['GROUP_ID']))
        log.info('Groups: {}'.format(groups))
        for group in groups:
            # Get observations in the group
            idx = np.where(self.obs_table['GROUP_ID'] == group)[0]
            obs_table_group = self.obs_table[idx]
            log.info('Processing group {} with {} observations'.format(group, len(obs_table_group)))

            # Build the model
            if modeltype == "3D":
                model = CubeBackgroundModel.define_cube_binning(obs_table_group, method='default')
                model.fill_obs(obs_table_group, self.data_store)
                model.smooth()
                model.compute_rate()
                self.models3D.append(model)

            elif modeltype == "2D":
                ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
                offset = sqrt_space(start=0, stop=2.5, num=100) * u.deg
                model = EnergyOffsetBackgroundModel(ebounds, offset)
                model.fill_obs(obs_table_group, self.data_store, self.excluded_sources)
                model.compute_rate()
                self.models2D.append(model)
            else:
                raise ValueError("Invalid model type: {}".format(modeltype))

    def save_model(self, modeltype, ngroup):
        """Save model to fits for one group in zenithal angle and efficiency.

        Parameters
        ----------
        modeltype : str
            type of the background modelisation: 3D or 2D
        ngroup : int
            group number

        """
        filename = self.outdir + '/background_{}_group_{:03d}_table.fits.gz'.format(modeltype, ngroup)
        if modeltype == "3D":
            self.models3D[ngroup].write(str(filename), format='table', clobber=True)
        if modeltype == "2D":
            self.models2D[ngroup].write(str(filename), overwrite=True)

    def save_models(self, modeltype):
        """Save model to fits for all the groups in zenithal angle and efficiency

        Parameters
        ----------
        modeltype : str
            type of the background modelisation: 3D or 2D

        """
        for ngroup in range(self.ntot_group):
            self.save_model(modeltype, ngroup)


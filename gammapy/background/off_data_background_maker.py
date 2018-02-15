# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
from astropy.table import join as table_join
from ..data import ObservationTable, ObservationGroupAxis, ObservationGroups
from .models import FOVCubeBackgroundModel
from .models import EnergyOffsetBackgroundModel
from ..utils.energy import EnergyBounds
from ..utils.nddata import sqrt_space

__all__ = [
    'OffDataBackgroundMaker',
]

log = logging.getLogger(__name__)


class OffDataBackgroundMaker(object):
    """OffDataBackgroundMaker class.

    Class that will select an OFF list run from a Data list and then group this runlist in group. Then for each
    group, it will compute the background rate model in 3D *(X, Y, energy)* or 2D *(energy, offset)* via the class
    `~gammapy.background.FOVCubeBackgroundModel` (3D) or `~gammapy.background.EnergyOffsetBackgroundModel` (2D).

    Parameters
    ----------
    data_store : `~gammapy.data.DataStore`
        Data for the background model
    outdir : str
        directory where will go the output
    run_list : str
        filename where is store the OFF run list
    obs_table : `~gammapy.data.ObservationTable`
        observation table of the OFF run List used for the background modelling
        require GROUP_ID column
    ntot_group : int
        Number of group
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

        self.obs_table_grouped_filename = self.outdir + '/obs.fits'
        self.group_table_filename = self.outdir + '/group-def.fits'

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
        obs_table.write(str(filename), overwrite=True)
        self.obs_table = obs_table

        filename = self.group_table_filename
        log.info('Writing {}'.format(filename))
        obs_groups.obs_groups_table.write(str(filename), overwrite=True)
        self.ntot_group = obs_groups.n_groups

    def make_model(self, modeltype, ebounds=None, offset=None):
        """Make background models.

        Create the list of background model (`~gammapy.background.FOVCubeBackgroundModel` (3D) or
        `~gammapy.background.EnergyOffsetBackgroundModel` (2D)) for each group

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        ebounds : `~gammapy.utils.energy.EnergyBounds`
            Energy bounds vector (1D)
        offset : `~astropy.coordinates.Angle`
            Offset vector (1D)
        """
        groups = sorted(np.unique(self.obs_table['GROUP_ID']))
        log.info('Groups: {}'.format(groups))
        for group in groups:
            # Get observations in the group
            idx = np.where(self.obs_table['GROUP_ID'] == group)[0]
            obs_table_group = self.obs_table[idx]
            obs_ids = list(obs_table_group['OBS_ID'])
            log.info('Processing group {} with {} observations'.format(group, len(obs_table_group)))

            # Build the model
            if modeltype == "3D":
                model = FOVCubeBackgroundModel.define_cube_binning(obs_table_group, method='default')
                model.fill_obs(obs_table_group, self.data_store)
                model.smooth()
                model.compute_rate()
                self.models3D[str(group)] = model
            elif modeltype == "2D":
                if not ebounds:
                    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 15, 'TeV')
                if not offset:
                    offset = sqrt_space(start=0, stop=2.5, num=100) * u.deg
                model = EnergyOffsetBackgroundModel(ebounds, offset)
                model.fill_obs(obs_ids=obs_ids, data_store=self.data_store, excluded_sources=self.excluded_sources)
                model.compute_rate()
                self.models2D[str(group)] = model
            else:
                raise ValueError("Invalid model type: {}".format(modeltype))

    @staticmethod
    def filename(modeltype, group_id, smooth=False):
        """Filename for a given ``modeltype`` and ``group_id``.

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        group_id : int
            number of the background model group
        smooth : bool
            True if you want to use the smooth bkg model
        """
        if smooth:
            return 'smooth_background_{}_group_{:03d}_table.fits.gz'.format(modeltype, group_id)
        else:
            return 'background_{}_group_{:03d}_table.fits.gz'.format(modeltype, group_id)

    def save_model(self, modeltype, ngroup, smooth=False):
        """Save model to fits for one group.

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        ngroup : int
            Groups ID
        """
        filename = self.outdir + "/" + self.filename(modeltype, ngroup, smooth)

        if modeltype == "3D":
            if str(ngroup) in self.models3D.keys():
                self.models3D[str(ngroup)].write(str(filename), format='table', overwrite=True)
            else:
                log.info("No run in the group {}".format(ngroup))
        if modeltype == "2D":
            if str(ngroup) in self.models2D.keys():
                self.models2D[str(ngroup)].write(str(filename), overwrite=True)
            else:
                log.info("No run in the group {}".format(ngroup))

    def save_models(self, modeltype, smooth=False):
        """Save model to fits for all the groups.

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        """
        for ngroup in range(self.ntot_group):
            self.save_model(modeltype, ngroup, smooth)

    def smooth_models(self, modeltype):
        """Smooth the bkg model for each group.

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        """
        for ngroup in range(self.ntot_group):
            self.smooth_model(modeltype, ngroup)

    def smooth_model(self, modeltype, ngroup):
        """Smooth the bkg model for one group.

        Parameters
        ----------
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        ngroup : int
            Groups ID
        """
        if modeltype == "3D":
            if str(ngroup) in self.models3D.keys():
                self.models3D[str(ngroup)].smooth()
            else:
                log.info("No run in the band {}".format(ngroup))
        if modeltype == "2D":
            if str(ngroup) in self.models2D.keys():
                self.models2D[str(ngroup)].smooth()
            else:
                log.info("No run in the band {}".format(ngroup))

    def make_bkg_index_table(self, data_store, modeltype, out_dir_background_model=None, filename_obs_group_table=None,
                             smooth=False):
        """Make background model index table.

        Parameters
        ----------
        data_store : `~gammapy.data.DataStore`
            `DataStore` for the runs for which ones we want to compute a background model
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        out_dir_background_model :  str
            directory where are located the backgrounds models for each group
        filename_obs_group_table : str
            name of the file containing the `~astropy.table.Table` with the group infos
        smooth : bool
            True if you want to use the smooth bkg model

        Returns
        -------
        index_table_bkg : `~astropy.table.Table`
            Index hdu table only for the background in order to associate a bkg model for each observation
        """
        obs_table = data_store.obs_table
        if not filename_obs_group_table:
            filename_obs_group_table = self.group_table_filename
        if not out_dir_background_model:
            out_dir_background_model = data_store.hdu_table.meta["BASE_DIR"]
        table_group = Table.read(filename_obs_group_table)
        axes = ObservationGroups.table_to_axes(table_group)
        groups = ObservationGroups(axes)
        obs_table = ObservationTable(obs_table)
        obs_table = groups.apply(obs_table)

        data = []
        for obs in obs_table:
            try:
                group_id = obs['GROUP_ID']
            except IndexError:
                log.warning('Found no GROUP_ID for {}'.format(obs["OBS_ID"]))
                continue
            row = dict()
            row["OBS_ID"] = obs["OBS_ID"]
            row["HDU_TYPE"] = "bkg"
            row["FILE_DIR"] = str(out_dir_background_model)
            row["FILE_NAME"] = self.filename(modeltype, group_id, smooth)
            if modeltype == "2D":
                row["HDU_NAME"] = "bkg_2d"
                row["HDU_CLASS"] = "bkg_2d"
            elif modeltype == '3D':
                row["HDU_NAME"] = "bkg_3d"
                row["HDU_CLASS"] = "bkg_3d"
            else:
                raise ValueError('Invalid modeltype: {}'.format(modeltype))
            data.append(row)

        index_table_bkg = Table(data)
        return index_table_bkg

    def make_total_index_table(self, data_store, modeltype, out_dir_background_model=None,
                               filename_obs_group_table=None, smooth=False):
        """Create a hdu-index table with a row containing the link to the background model for each observation.

        Parameters
        ----------
        data_store : `~gammapy.data.DataStore`
             `DataStore` for the runs for which ones we want to compute a background model
        modeltype : {'3D', '2D'}
            Type of the background modelisation
        out_dir_background_model :  str
            directory where are located the backgrounds models for each group
        filename_obs_group_table : str
            name of the file containing the `~astropy.table.Table` with the group infos
        smooth : bool
            True if you want to use the smooth bkg model

        Returns
        -------
        index_table_new : `~astropy.table.Table`
            Index hdu table with a background row
        """
        index_table_bkg = self.make_bkg_index_table(data_store, modeltype, out_dir_background_model,
                                                    filename_obs_group_table, smooth)
        index_bkg = np.where(data_store.hdu_table["HDU_CLASS"] == "bkg_3d")[0].tolist()
        data_store.hdu_table.remove_rows(index_bkg)
        index_table_new = vstack([data_store.hdu_table, index_table_bkg])
        return index_table_new

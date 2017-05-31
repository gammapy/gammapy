# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table
import numpy as np
from ...utils.testing import requires_dependency, requires_data
from ...data import ObservationTable, DataStore
from ..models import EnergyOffsetBackgroundModel
from ..off_data_background_maker import OffDataBackgroundMaker


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_model(tmpdir):
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    runlist_filename = str(tmpdir.join('run.lis'))
    bgmaker = OffDataBackgroundMaker(data_store, outdir=str(tmpdir), run_list=runlist_filename)

    bgmaker.select_observations(selection='all')
    table = Table.read(runlist_filename, format='ascii.csv')
    assert table['OBS_ID'][1] == 23526

    bgmaker.group_observations()
    table = ObservationTable.read(str(tmpdir / 'obs.fits'))
    assert list(table['GROUP_ID']) == [0, 0, 0, 1]
    table = ObservationTable.read(str(tmpdir / 'group-def.fits'))
    assert list(table['ZEN_PNT_MAX']) == [49, 90]

    # TODO: Fix 3D code
    # bgmaker.make_model("3D")
    # bgmaker.save_models("3D")
    # model = FOVCubeBackgroundModel.read(str(tmpdir / 'background_3D_group_001_table.fits.gz'))
    # assert model.counts_cube.data.sum() == 1527

    bgmaker.make_model("2D")
    bgmaker.save_models("2D")
    model = EnergyOffsetBackgroundModel.read(str(tmpdir / 'background_2D_group_001_table.fits.gz'))
    assert model.counts.data.value.sum() == 1398

    index_table_new = bgmaker.make_total_index_table(data_store, "2D", None, None)
    table_bkg = index_table_new[np.where(index_table_new["HDU_NAME"] == "bkg_2d")]

    name_bkg_run023523 = table_bkg[np.where(table_bkg["OBS_ID"] == 23523)]["FILE_NAME"]
    assert str(tmpdir) + "/" + name_bkg_run023523[0] == str(tmpdir) + '/background_2D_group_001_table.fits.gz'

    name_bkg_run023526 = table_bkg[np.where(table_bkg["OBS_ID"] == 23526)]["FILE_NAME"]
    assert str(tmpdir) + "/" + name_bkg_run023526[0] == str(tmpdir) + '/background_2D_group_000_table.fits.gz'

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table
from ...utils.testing import requires_dependency, requires_data
from ...data import ObservationTable, DataStore
from ..models import CubeBackgroundModel, EnergyOffsetBackgroundModel
from ..off_data_background_maker import OffDataBackgroundMaker
from ...datasets import gammapy_extra
import os


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_model(tmpdir):
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')

    bgmaker = OffDataBackgroundMaker(data_store, outdir=tmpdir)

    bgmaker.select_observations(selection='all')
    table = Table.read('run.lis', format='ascii.csv')
    assert table['OBS_ID'][1] == 23526

    bgmaker.group_observations()
    table = ObservationTable.read(str(tmpdir / 'obs.ecsv'), format='ascii.ecsv')
    assert list(table['GROUP_ID']) == [0, 0, 0, 1]
    table = ObservationTable.read(str(tmpdir / 'group-def.ecsv'), format='ascii.ecsv')
    assert list(table['ZEN_PNT_MAX']) == [49, 90]

    # TODO: Fix 3D code
    # bgmaker.make_model("3D")
    # bgmaker.save_models("3D")
    # model = CubeBackgroundModel.read(str(tmpdir / 'background_3D_group_001_table.fits.gz'))
    # assert model.counts_cube.data.sum() == 1527

    bgmaker.make_model("2D")
    bgmaker.save_models("2D")
    model = EnergyOffsetBackgroundModel.read(str(tmpdir / 'background_2D_group_001_table.fits.gz'))
    assert model.counts.data.value.sum() == 1398

    directory = str(gammapy_extra.dir) + '/datasets/hess-crab4-hd-hap-prod2/'
    bgmaker.background_symlinks(data_store.obs_table, str(tmpdir), directory, "2D", None)
    run023523 = os.readlink(directory + "run023400-023599/run023523/background_023523.fits.gz")
    assert run023523 == str(tmpdir) + '/background_2D_group_001_table.fits.gz'
    run023526 = os.readlink(directory + "run023400-023599/run023526/background_023526.fits.gz")
    assert run023526 == str(tmpdir) + '/background_2D_group_000_table.fits.gz'

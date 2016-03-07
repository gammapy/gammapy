"""
Example script for the new class OffDataBackgroundMaker
"""
from astropy.table import Table
from gammapy.background import CubeBackgroundModel
from gammapy.background import EnergyOffsetBackgroundModel
from gammapy.data import ObservationTable

from gammapy.data import DataStore
from gammapy.background.off_data_background_maker import OffDataBackgroundMaker

def test_background_model():
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    out_dir = "out"
    bgmaker = OffDataBackgroundMaker(data_store, out_dir)

    selection= "debug"
    bgmaker.select_observations(selection)
    table = Table.read('run.lis', format='ascii.csv')
    assert table['OBS_ID'][1] == 23526


    run_list="run.lis"
    bgmaker.group_observations()
    table = ObservationTable.read('out/obs.ecsv', format='ascii.ecsv')
    assert list(table['GROUP_ID']) == [0, 0, 0, 1]
    table = ObservationTable.read('out/group-def.ecsv', format='ascii.ecsv')
    assert list(table['ZEN_PNT_MAX']) == [49, 90]

    bgmaker.make_model("3D")
    model = CubeBackgroundModel.read('out/background_3D_group_001_table.fits.gz')
    assert model.counts_cube.data.sum() == 1527

    bgmaker.make_model("2D")
    model = EnergyOffsetBackgroundModel.read('out/background_2D_group_001_table.fits.gz')
    assert model.counts.data.value.sum() == 1398

    """
    bgmaker.save()
    """
if __name__ == '__main__':
    test_background_model()



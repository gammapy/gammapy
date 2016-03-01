# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from ...background import CubeBackgroundModel
from ...background import EnergyOffsetBackgroundModel
from ...utils.testing import requires_data, requires_dependency, run_cli
from ...datasets import gammapy_extra
from ...data import ObservationTable
from ..background_model import background_cli


def test_background_cli_help():
    exc = run_cli(background_cli, ['--help'])

    assert exc.value.args[0] == 0

    # assert 'Background models' in result.output


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_cli_all(tmpdir):
    data_dir = gammapy_extra.filename('datasets/hess-crab4-hd-hap-prod2/')
    out_dir = 'out'
    cmds = [data_dir, out_dir]

    # Step 1: list
    run_cli(background_cli, cmds + ['list', '--selection', 'debug'])
    table = Table.read('runs.lis', format='ascii.csv')
    assert table['OBS_ID'][1] == 23526

    # Step 2: group
    run_cli(background_cli, cmds + ['group'])
    table = ObservationTable.read('out/obs.ecsv', format='ascii.ecsv')
    assert list(table['GROUP_ID']) == [0, 0, 0, 1]
    table = ObservationTable.read('out/group-def.ecsv', format='ascii.ecsv')
    assert list(table['ZEN_PNT_MAX']) == [49, 90]

    # Step 3: model 3D
    run_cli(background_cli, cmds + ['model', '--modeltype', '3D'])
    model = CubeBackgroundModel.read('out/background_3D_group_001_table.fits.gz')
    assert model.counts_cube.data.sum() == 1527

    # Step 4: model 2D
    run_cli(background_cli, cmds + ['model', '--modeltype', '2D'])
    model = EnergyOffsetBackgroundModel.read('out/background_2D_group_001_table.fits.gz')
    assert model.counts.data.value.sum() == 1398

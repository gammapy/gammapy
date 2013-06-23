from ...obs import RunList
from ..maps import Maps
from ..reflected import ReflectedRegionMaker

"""
class TestReflectedBgMaker(unittest.TestCase):

    @unittest.skip('TODO')
    def test_analysis(self):
        runs = RunList('runs.lis')
        maps = Maps('maps.fits')
        reflected_bg_maker = ReflectedBgMaker(runs, maps, psi=2, theta=0.1)
        total_maps = Maps('total_maps.fits')
        for run in runs:
            run_map = total_maps.cutout(run)
            reflected_bg_maker.make_n_reflected_map(run, run_map)
            total_maps.add(run_map)
        total_maps.save('n_reflected.fits')
"""

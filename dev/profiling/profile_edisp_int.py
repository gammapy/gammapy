import cProfile
import pstats
from gammapy.irf import EnergyDispersion2D
from gammapy.utils.energy import EnergyBounds
from gammapy.data import DataStore
import astropy.units as u
import numpy as np

# Copied from gp-extra/notebooks/data-iact

data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
obs = data_store.obs(obs_id=23523)
edisp = obs.edisp
cProfile.run('edisp.peek()', 'restats')
p = pstats.Stats('restats')
p.sort_stats('cumtime').print_stats(10)
#p.sort_stats('cumtime').print_stats(20)
#p.strip_dirs().sort_stats('cumtime').print_callers(20)


#filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_edisp_2d_023523.fits.gz'

#edisp2d = EnergyDispersion2D.read(filename)
#print(edisp2d)

#cProfile.run('edisp2d.peek()', 'restats')
#p = pstats.Stats('restats')
#p.sort_stats('cumtime').print_stats(20)
#p.strip_dirs().sort_stats('cumtime').print_callers(20)

#edisp2d.peek()


#offset = 1.2 * u.deg
#e_true = 1 * u.TeV
#e_reco = EnergyBounds.equal_log_spacing(100 * u.GeV, 10 * u.TeV, 100)
#cProfile.run('c = edisp2d.get_response(offset=offset, e_true=e_true, e_reco=e_reco)', 'restats')
#p = pstats.Stats('restats')
#p.sort_stats('cumtime').print_stats(20)
#p.strip_dirs().sort_stats('cumtime').print_callers(20)

#print(c)
#idx = np.where(np.nonzero(c))[0]
#print(len(idx))
#resp = edisp2d.get_response(offset=offset, e_true=e_true, e_reco=e_reco)
#print(resp)
#edisp = edisp2d.to_energy_dispersion(offset=offset)  
#print(edisp)

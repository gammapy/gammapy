"""Test spectrum energy grouping.
"""
import warnings
import astropy.units as u
from gammapy.spectrum import SpectrumObservation, SpectrumEnergyGroupMaker

warnings.filterwarnings('ignore')

filename = '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits'
obs = SpectrumObservation.read(filename)
table = obs.stats_table()

seg = SpectrumEnergyGroupMaker(obs=obs)
ebounds = [0.03, 1, 3, 10, 30] * u.TeV
seg.compute_range_safe()
seg.compute_groups_fixed(ebounds=ebounds)

print('\nTable of original energy bins:')
seg.table[['energy_group_idx', 'bin_idx', 'energy_min', 'energy_max']].pprint(max_lines=-1)

print('\nTable of grouped energy bins:')
seg.groups.to_group_table().pprint(max_lines=-1)

print('\nTable of grouped energy bins:')
seg.groups.to_total_table().pprint(max_lines=-1)

print(seg)

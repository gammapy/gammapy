"""Starting with a list of observations, find the extreme ones for test purposes.

It uses the runinfo.fits file from the H.E.S.S. fits exporter.
See instructions on how to get it here:
 https://hess-confluence.desy.de/confluence/display/HESS/HESS+Open+Source+Tools+-+HOWTO+download+and+organise+HESS+FITS+data

It finds the observations with min, max:

- Altitude
- Muon efficiency

both for 3 and 4 telescope observations: 8 obs in total.
"""
from gammapy.data import DataStoreIndexTable


# get full list of H.E.S.S. observations
filename = '/Users/deil/work/_Data/hess/fits/parisanalysis/fits_prod02/pa/Model_Deconvoluted_Prod26/Mpp_Std/runinfo.fits'
obs_table = DataStoreIndexTable.read(filename)


# define empty obs table to store the extreme obs
obs_table_extreme = obs_table[0:0]

# loop over ntels
ntels = [3, 4]
for i_ntels in ntels:
    select_tels = dict(type='par_box', variable='N_TELS',
                       value_range=[i_ntels, i_ntels + 1])
    observation_table_tels = obs_table.select_observations(select_tels)

    # loop over colnames
    colnames = ['ALT_PNT', 'MUONEFF']
    # print(observation_table.colnames)
    for i_col in colnames:
        # sort table
        observation_table_tels.sort(i_col)

        # get extreme entries
        obs_table_extreme.add_row(observation_table_tels[0])
        obs_table_extreme.add_row(observation_table_tels[-1])

# show extreme table
print()
print("extreme observation table")
print(obs_table_extreme.meta)
print(obs_table_extreme)

# save extreme obs table to fits file
filename = 'extreme_observation_list.fits'
print("\nWriting {}.".format(filename))
obs_table_extreme.write(filename, overwrite=True)

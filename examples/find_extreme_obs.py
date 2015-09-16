"""Starting with a list of observatons, find the extreme ones for test purposes.

It uses the runinfo.fits file from the H.E.S.S. fits exporter.
See instructions on how to get it here:
 https://hess-confluence.desy.de/confluence/display/HESS/HESS+Open+Source+Tools+-+HOWTO+download+and+organise+HESS+FITS+data

It finds the observations with min, max:

- Altitude
- Muon efficiency

both for 3 and 4 telescope observations: 8 obs in total.

Use this in your terminal o get a list of available options:
 python find_extreme_obs.py -h
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals) # python 2 as python 3
import argparse
from gammapy.obs import DataStore

DEBUG = 0

HESSFITS_MPP = '/home/mapaz/astropy/gammapy_tutorial/HESS_fits_data/pa/Model_Deconvoluted_Prod26/Mpp_Std'
OUTFILE = 'extreme_observation_list.fits'

def find_extreme_observations(fits_path, outfile, overwrite):
    """Make total observation list and select the observations.

    Parameters
    ----------
    fits_path : str
        Path to dir containing input observation list.
    outfile : str
        Name of output observation list file.
    overwrite : bool
        Flag to overwrite output file.
    """
    if DEBUG:
        print()
        print("#######################################")
        print("# Starting create_bg_observation_list #")
        print("#######################################")

    # get full list of H.E.S.S. observations
    data_store = DataStore(dir=fits_path)
    observation_table = data_store.make_observation_table()
    if DEBUG:
        print()
        print("full observation table")
        print(observation_table)

    # TODO: the column format is not the accepted format in Gammapy!!!

    # define empty obs table to store the extreme obs
    obs_table_extreme = observation_table[0:0]
    if DEBUG:
        print()
        print("extreme observation table (empty)")
        print(obs_table_extreme)

    # loop over ntels
    ntels = [3, 4]
    for i_ntels in ntels:
        select_tels = dict(type='par_box', variable='N_TELS',
                           value_range=[i_ntels, i_ntels + 1])
        observation_table_tels = observation_table.select_observations(select_tels)
        if DEBUG:
            print()
            print("observation table {} tels".format(i_ntels))
            print(observation_table_tels)

        # loop over colnames
        colnames = ['ALT_PNT', 'MUONEFF']
        for i_col in colnames:
            # sort table
            observation_table_tels.sort(i_col)
            if DEBUG:
                print()
                print("observation table {} tels sorted in {}".format(i_ntels, i_col))
                print(observation_table_tels)

            # get extreme entries
            obs_table_extreme.add_row(observation_table_tels[0])
            obs_table_extreme.add_row(observation_table_tels[-1])

    # show extreme table
    print()
    print("extreme observation table")
    print(obs_table_extreme.meta)
    print(obs_table_extreme)

    # save extreme obs table to fits file
    if OUTFILE != '':
        print()
        print("Writing {}.".format(outfile))
        obs_table_extreme.write(outfile, overwrite=overwrite)


def main(args=None):
    """Main function: parse arguments and launch the whole analysis chain.
    """
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--fitspath', type=str,
                        default=HESSFITS_MPP,
                        help='Dir path to input event list fits files '
                        '(default is {}).'.format(HESSFITS_MPP))
    parser.add_argument('--outfile', type=str,
                        default=OUTFILE,
                        help='Output file name (default is {}).'.format(OUTFILE))
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')

    args = parser.parse_args()

    find_extreme_observations(args.fitspath, args.outfile, args.overwrite)


if __name__ == '__main__':
    main()

"""Combine all Crab flux points into a single file
and change columns to energy and flux.

As input copy the `crab` folder with `*.ipac` files
from Rolf Buehler into the working directory.

Summary of output:

$ stilts tpipe crab_mwl.fits omode=stats
Total Rows: 437
+--------------------+---------------+--------------+--------------+-------------+------+
| column             | mean          | stdDev       | min          | max         | good |
+--------------------+---------------+--------------+--------------+-------------+------+
| energy             | 698762.06     | 4776016.5    | 2.928466E-13 | 7.4997296E7 | 437  |
| energy_flux        | -0.002919502  | 0.06096109   | -1.2758244   | 3.47928E-8  | 437  |
| energy_flux_err_lo | 3.2466577E-10 | 6.804524E-10 | 0.0          | 3.3667E-9   | 393  |
| energy_flux_err_hi | 3.2521535E-10 | 6.803111E-10 | 0.0          | 3.3666E-9   | 393  |
| energy_flux_err    | 3.2494055E-10 | 6.803782E-10 | 0.0          | 3.36665E-9  | 393  |
| component          |               |              |              |             | 437  |
| paper              |               |              |              |             | 437  |
+--------------------+---------------+--------------+--------------+-------------+------+

$ gzip crab_mwl.fits
$ cp crab_mwl.fits.gz ../gammapy/datasets/data/tev_spectra/

"""
from glob import glob
import numpy as np
from astropy import table
from astropy.table import Table, vstack
import astropy.units as u
from astropy.units import Quantity


def get_combined_table():
    """Combine information from all tables in a single table
    (they already all have the same column names, unit, format)
    """
    filenames = glob('crab/*/*.ipac')
    
    tables = []
    for filename in filenames:    
        table = Table.read(filename, format='ascii.ipac')
        tokens = filename.split('/')
        table['component'] = tokens[-2]
        table['paper'] = tokens[-1].replace('_vFv.ipac', '')
        tables.append(table)
    
    table = vstack(tables)
    print('INFO: Found {0} files with {1} flux points.'.format(len(filenames), len(table)))

    return table


def remove_flare_data(table):
    """Most users will not need the Crab flare SED measurements
    and they have lots of upper limits ... remove for now.
    """
    papers = ['agile_sept2007', 'chandra_april2011',
              'fermi_7max_april11', 'fermi_bb4_march2013',
              'fermi_feb2009', 'fermi_sept2010']
    mask = np.array([paper not in papers for paper in table['paper']])
    print('WARNING: Removing {0} datasets with {1} flux points.'
          ''.format(len(papers), np.invert(mask).sum()))
    table = table[mask]

    # We don't care for these columns
    columns = ['TS', 'Npred', 'ulim_vFv']
    print('WARNING: Removing columns: {0}'.format(columns))
    table.remove_columns(columns)

    return table


def combine_columns(table):
    """Combine vFv error information.
    
    At the moment it's scattered across multiple columns.
    
    We want the `ed_vFv` and `eu_vFv` columns to be filled for all rows.
    """
    mask = table['ed_vFv'].mask
    table['ed_vFv'][mask] = table['vFv'][mask] - table['d_vFv'][mask]
    
    mask = table['ed_vFv'].mask
    table['ed_vFv'][mask] = table['e_vFv'][mask]
    
    mask = table['eu_vFv'].mask
    table['eu_vFv'][mask] = table['u_vFv'][mask] - table['vFv'][mask]
    
    mask = table['eu_vFv'].mask
    table['eu_vFv'][mask] = table['e_vFv'][mask]

    # TODO: I'm getting weird results for the following attempts
    # to remove or mask incorrect data.
    # It's either bugs in Table mask handling or a lack of understanding
    # how it works on my part ... don't have time to investigate now.

    # veron_vFv.ipac should have had missing value entries
    mask = table['paper'].filled('a') == 'veron'
    table['ed_vFv'].mask[mask] = True
    table['eu_vFv'].mask[mask] = True

    # Remove upper limits
    #import IPython; IPython.embed()
    #mask = table['eu_vFv'].filled(1) == 0
    #print('WARNING: Removing {0} rows with missing `eu_vFv`:\n{1}'
    #      ''.format(mask.sum(), table[mask]['paper']))
    #table = table[~mask]

    return table


def clean_up(table_in):
    """Create a new table with exactly the columns / info we want.
    """
    table = Table()
    
    v = Quantity(table_in['v'].data, table_in['v'].unit)
    energy = v.to('MeV', equivalencies=u.spectral())
    table['energy'] = energy
    
    #vFv = Quantity(table['vFv'].data, table['vFv'].unit)
    #flux = (vFv / (energy ** 2)).to('cm^-2 s^-1 MeV^-1')
    
    table['energy_flux'] = table_in['vFv']
    table['energy_flux_err_lo'] = table_in['ed_vFv'] 
    table['energy_flux_err_hi'] = table_in['eu_vFv']
    # Compute symmetrical error because most chi^2 fitters
    # can't handle asymmetrical Gaussian erros
    table['energy_flux_err'] = 0.5 * (table_in['eu_vFv'] + table_in['ed_vFv'])

    table['component'] = table_in['component']
    table['paper'] = table_in['paper']
    
    mask = table['energy_flux_err_hi'] == 0

    return table


def table_info(table, label):
    # Print basic info to console
    print('\n*** {0} ***'.format(label))
    print('Rows: {0}'.format(len(table)))
    print('Columns: {0}'.format(table.colnames))
    print('')

    # Save to file for debugging
    filename = 'crab_mwl_{0}.ipac'.format(label)
    print('INFO: Writing {0}'.format(filename))
    table.write(filename, format='ascii.ipac')


if __name__ == '__main__':
    table = get_combined_table()
    table_info(table, 'get_combined_table')

    table = remove_flare_data(table)
    table_info(table, 'remove_flare_data')

    table = combine_columns(table)
    table_info(table, 'combine_columns')

    table = clean_up(table)
    table_info(table, 'clean_up')

    filename = 'crab_mwl.fits'
    print('INFO: Writing {0}'.format(filename))
    table.write(filename, overwrite=True) 

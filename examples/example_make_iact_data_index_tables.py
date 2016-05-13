"""Example how to construct index tables.

* Observation index: http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html
* HDU index: http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html
* Dataset examples:
  * https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4-hd-hap-prod2
  * https://github.com/gammapy/gammapy-extra/tree/master/datasets/hess-crab4-pa

TODO: clean this up and make it part of the docs!
"""
from astropy.io import fits
from astropy.table import Table

# Write minimal files and observations table needed for the gammapy data store

obs_ids = [23523, 23526, 23559, 23592]
file_types = ['events', 'aeff', 'edisp', 'psf', 'background']

# FILE TABLE

rows = []
for obs in obs_ids:
    for filetype in file_types:
        #name = "hess_" + filetype + "_" + obs + ".fits.gz"
        name = "hess_{}_{:06d}.fits.gz".format(filetype, obs)
        if filetype == 'events':
        #    name = "hess_events_simulated_" + obs + ".fits"
            name = "hess_events_simulated_{:06d}.fits".format(obs)
        if filetype == 'background':
        #    name = "hess_bkg_offruns_" + obs + ".fits.gz"
            name = "hess_bkg_offruns_{:06d}.fits.gz".format(obs)
        data = dict()
        data['OBS_ID'] = obs
        data['TYPE'] = filetype
        data['NAME'] = name
        data['SIZE'] = ''
        data['MTIME'] = ''
        data['MD5'] = ''

        hdu_list = fits.open(str(name))
        hdu = hdu_list[1]
        header = hdu.header
        data['HDUNAME'] = hdu.name
        if 'HDUCLAS2' in header:
            data['HDUCLASS'] = header['HDUCLAS2']
        else:
            data['HDUCLASS'] = 'EVENTS'
        rows.append(data)

names = [
    'OBS_ID', 'TYPE',
    'NAME', 'SIZE', 'MTIME', 'MD5',
    'HDUNAME', 'HDUCLASS',
    ]
table = Table(rows=rows, names=names)
table.write("files.fits.gz", overwrite=True)



# OBS TABLE

rows = []
for obs in obs_ids:

    data = dict()
    #data['OBS_ID'] = obs
    data['OBS_ID'] = obs
    rows.append(data)

table = Table(rows=rows)
table.meta['MJDREFI'] = 51544
table.meta['MJDREFF'] = 0.5
table.write("observations.fits.gz", overwrite=True)


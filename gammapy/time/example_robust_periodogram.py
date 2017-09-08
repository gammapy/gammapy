import numpy as np
from astropy.table import Table

from .robust_periodogram import robust_periodogram
from .plot_periodogram import plot_periodogram

table = Table.read('https://github.com/gammapy/gamma-cat/raw/master/input/data/2006/2006A%2526A...460..743A/tev-000119-lc.ecsv', format='ascii.ecsv')
time = (table['time_max'].data + table['time_min'].data) / 2
flux = table['flux'].data
flux_err = table['flux_err'].data

result = robust_periodogram(time, flux*10**12, flux_err*10**12, 0.001, 'cauchy', 1, 10, 'None')
plot_periodogram(time, flux, flux_err, result['pgrid'],
        result['psd'], result['swf'], result['period'],
        result['significance']
        )

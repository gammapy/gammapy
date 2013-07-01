"""Test how well FITS supports 1e2 to 1e4 HDUs,
i.e. if accessing them is fast.

On my Macbook (with an SSD) it takes about
1.2 sec to open a FITS file with 1000 HDUs
and this scales perfectly linearly with the number of HDUs.

Opening with cfitsio (using ftlist test.fits H) is about
10 x faster.
"""
import logging
logging.basicConfig(level=logging.INFO)
from time import time
import sys
import numpy as np
from astropy.io import fits

def write_test_file(filename, n_hdu, n_bytes):
    hdus = fits.HDUList()
    data = np.zeros(n_bytes / 8, dtype='float64')
    for hdu_nr in range(n_hdu):
        hdu_name = 'H_{0:06d}'.format(hdu_nr)
        hdu = fits.ImageHDU(data, name=hdu_name)
        hdus.append(hdu)
    hdus.writeto(filename, clobber=True)

if __name__ == '__main__':
    n_hdu = int(sys.argv[1])
    try:
        n_bytes = int(sys.argv[2])
    except:
        n_bytes = 1

    logging.info('n_hdu = {0}'.format(n_hdu))
    logging.info('n_bytes = {0}'.format(n_bytes))
    
    filename = 'test.fits'
    logging.info('Writing {0}'.format(filename))
    write_test_file(filename, n_hdu, n_bytes)
    
    t = time()
    logging.info('Reading {0}'.format(filename))
    fits.open(filename)
    t = time() - t
    logging.info('Opening took {0} sec'.format(t))
    
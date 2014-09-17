"""Combine all Fermi IRF data into a single file.
"""
from glob import glob
import numpy as np
from astropy import table
from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as u
from astropy.units import Quantity


filenames = glob('*.txt')

layers = []
primary = fits.PrimaryHDU()
layers.append(primary)
for filename in filenames:    
    print(filename)
    table = Table.read(filename, format='ascii')
    table.write('temp.fits', overwrite=True)
    hdu = fits.open('temp.fits')[1]
    layers.append(hdu)
hdus = fits.HDUList(layers)
hdus.writeto('fermi_irf_data.fits', clobber=True) 


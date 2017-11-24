"""
Example file how to create fits file using gammapy.irf classes.

"""

import numpy as np
import astropy.units as u
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D, Background3D
from collections import OrderedDict
from astropy.io import fits
from gammapy.utils.fits import table_to_fits_table


provenance = OrderedDict([('ORIGIN', 'IRAP'),
                          ('DATE', '2017-09-27T12:02:24'),
                          ('TELESCOP', 'CTA'),
                          ('INSTRUME', 'PROD3B'),
                          ('ETC', 'ETC')])

# Set up some example data
energy = np.logspace(0, 1, 11) * u.TeV
energy_lo = energy[:-1]
energy_hi = energy[1:]
offset = np.linspace(0, 1, 4) * u.deg
offset_lo = offset[:-1]
offset_hi = offset[1:]
migra = np.linspace(0,3,4)
migra_lo = migra[:-1]
migra_hi = migra[1:]
detx = np.linspace(-6,6,11) * u.deg
detx_lo = detx[:-1]
detx_hi = detx[1:]
dety = np.linspace(-6,6,11) * u.deg
dety_lo = dety[:-1]
dety_hi = dety[1:]
aeff_data = np.ones(shape=(10,3))*u.cm*u.cm
edisp_data = np.ones(shape=(10, 3, 3))
bkg_data = np.ones(shape=(10,10,10)) / u.MeV / u.s / u.sr


# Create IRF Class objects with data
aeff = EffectiveAreaTable2D(energy_lo=energy_lo, energy_hi=energy_hi,
                            offset_lo=offset_lo, offset_hi=offset_hi,
                            data=aeff_data)

edisp = EnergyDispersion2D(e_true_lo=energy_lo, e_true_hi=energy_hi,
                           migra_lo=migra_lo, migra_hi=migra_hi,
                           offset_lo=offset_lo, offset_hi=offset_hi,
                           data=edisp_data)

bkg = Background3D(energy_lo=energy_lo, energy_hi=energy_hi,
                   detx_lo=detx_lo, detx_hi=detx_hi,
                   dety_lo=dety_lo, dety_hi=dety_hi,
                   data=bkg_data)

# Convert to astropy Table objects
table_aeff = aeff.to_table()
table_edisp = edisp.to_table()
table_bkg = bkg.to_table()

# Add any information that needs to be in the fits header
table_aeff.meta.update(provenance)
table_edisp.meta.update(provenance)
table_bkg.meta.update(provenance)

# Convert to fits HDU objects
hdu_aeff = table_to_fits_table(table_aeff, name='EFFECTIVE AREA')
hdu_edisp = table_to_fits_table(table_edisp, name='ENERGY DISPERSION')
hdu_bkg = table_to_fits_table(table_bkg, name='BACKGROUND')
prim_hdu = fits.PrimaryHDU()

# Alternatively, HDU can be obtained directly:
# hdu_aeff = aeff.to_fits(name='EFFECTIVE AREA')
# ...


fits.HDUList([prim_hdu, hdu_aeff, hdu_edisp, hdu_bkg]).writeto('irf_test.fits')

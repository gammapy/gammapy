from astropy.io import fits
from gammapy.datasets import gammapy_extra

filename = gammapy_extra.filename('datasets/catalogs/fermi/gll_psch_v08.fit.gz')
hdu = fits.open(filename)['Count Map']

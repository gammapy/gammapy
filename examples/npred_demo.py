from gammapy.datasets import FermiGalacticCenter
from astropy.io import fits
from gammapy.image import cube_to_image
from gammapy.spectral_cube import compute_npred_cube
FermiGalacticCenter.filenames()['exposure_cube']
exposure = [fits.open(FermiGalacticCenter.filenames()['exposure_cube'])[0], fits.open(FermiGalacticCenter.filenames()['exposure_cube'])[1]]
diffuse = [fits.open(FermiGalacticCenter.filenames()['diffuse_model'])[0], fits.open(FermiGalacticCenter.filenames()['diffuse_model'])[1]]
#print(repr(exposure.data))
#print(exposure.header)
#print(diffuse.header)
counts = fits.open(FermiGalacticCenter.filenames()['counts'])[1]
#ref = cube_to_image(exposure[0])
#lon, lat = coordinates(ref)
#lon = Quantity(lon, 'rad')
#lat = Quantity(lat, 'rad')
from astropy.units import Quantity
energy = Quantity(10000, 'MeV')
# This should be the other way round; failure may be due to extrapolation of some layers??
a = compute_npred_cube(diffuse, exposure)
#print(a)
# energy axis in diffuse and exposure cube are not the same format in HDU header
#cube = GammaSpectralCube.read_hdu(exposure)
#array = cube.flux(lat, lon, energy*0.1)
#print(a)
#print(a.shape)
print(a.sum())
hdu_image = fits.PrimaryHDU(data = a, header = diffuse[0].header)
hdu_energies = fits.TableHDU(data = diffuse[1].data, header = diffuse[1].header)
hdu_list = fits.HDUList(hdus=[hdu_image, hdu_energies])
hdu_list.writeto('test.fits', clobber=True)
print(counts.data.sum())
# The energy axis data in the fermi diffuse model header appears to be incorrect. Data from the table in the
# second [1] hdu table should be used instead...
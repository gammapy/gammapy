"""Plot cube background model and store it in fits.

The 'image' format file can be viewed with ds9.
"""
import matplotlib.pyplot as plt
from astropy.units import Quantity
from astropy.coordinates import Angle
from gammapy.background import FOVCube
from gammapy.datasets import gammapy_extra

filename = gammapy_extra.filename('test_datasets/background/bg_cube_model_test1.fits')
bg_cube_model = FOVCube.read(filename, format='table', scheme='bg_cube')

bg_cube_model.plot_image(energy=Quantity(2., 'TeV'))
bg_cube_model.plot_spectrum(coord=Angle([0., 0.], 'degree'))

outname = 'cube_background_model'
bg_cube_model.write('{}_bin_table.fits'.format(outname), format='table', clobber=True)
bg_cube_model.write('{}_image.fits'.format(outname), format='image', clobber=True)

plt.show()

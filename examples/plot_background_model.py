"""Plot background model and store as cube so that it can viewed with ds9.
"""
import matplotlib.pyplot as plt
from astropy.units import Quantity
from astropy.coordinates import Angle
from gammapy.background import CubeBackgroundModel
from gammapy import datasets

# read
filename = '../test_datasets/background/bg_cube_model_test.fits'
filename = datasets.get_path(filename, location='remote')
bg_model = CubeBackgroundModel.read(filename, format='bin_table')

# plot
bg_model.plot_image(energy=Quantity(2., 'TeV'))
bg_model.plot_spectrum(det=Angle([0., 0.], 'degree'))

# write
outname = 'cube_background_model'
bg_model.write('{}_bin_table.fits'.format(outname), format='bin_table',
               write_kwargs=dict(clobber=True))
bg_model.write('{}_image.fits'.format(outname), format='image',
               write_kwargs=dict(clobber=True))

plt.show()

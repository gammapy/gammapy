"""Plot background model and store as cube so that it can viewed with ds9.
"""
from astropy.units import Quantity
from astropy.coordinates import Angle
from gammapy.background import CubeBackgroundModel
from gammapy import datasets

filename = '../test_datasets/background/bg_cube_model_test.fits'
filename = datasets.get_path(filename, location='remote')
bg_model = CubeBackgroundModel.read_bin_table(filename)
#print("Plotting all images and spectra (can take a couple of mins).")
#bg_model.plot_images() # activate for all plots for all energy bins
#bg_model.plot_spectra() # activate for all plots for all det bins
bg_model.plot_images(energy=Quantity(2., 'TeV'))
bg_model.plot_spectra(det=Angle([0., 0.], 'degree'))
outname = 'cube_background_model'
bg_model.write_bin_table('{}_bin_table.fits'.format(outname), clobber=True)
bg_model.write_image('{}_image.fits'.format(outname), clobber=True)

"""Plot background model and store as cube so that it can viewed with ds9.
"""
from gammapy.background.models import CubeBackgroundModel
from gammapy import datasets

def plot_example():
    filename = '../test_datasets/background/bg_cube_model_test.fits'
    filename = datasets.get_path(filename, location='remote')
    bg_model = CubeBackgroundModel.read_bin_table(filename)
    print("Plotting all images and spectra (can take a couple of mins).")
    bg_model.plot_images()
    bg_model.plot_spectra()
    outname = 'cube_background_model'
    bg_model.write_bin_table('{}_bin_table.fits'.format(outname), clobber=True)
    bg_model.write_image('{}_image.fits'.format(outname), clobber=True)

if __name__ == '__main__':
    plot_example()

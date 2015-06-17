"""Plot background model and store as cube so that it can viewed with ds9.
"""
from gammapy.background.models import CubeBackgroundModel

def plot_example():
    #DIR = '/Users/deil/work/_Data/hess/HESSFITS/pa/Model_Deconvoluted_Prod26/Mpp_Std/background/'
    DIR = '/home/mapaz/astropy/testing_cube_bg_michael_mayer/background/'
    filename = DIR + 'hist_alt3_az0.fits.gz'
    bg_model = CubeBackgroundModel.read(filename)
    bg_model.plot_images('cube_background_model.png')
    bg_model.write_cube('cube_background_model.fits')

if __name__ == '__main__':
    plot_example()

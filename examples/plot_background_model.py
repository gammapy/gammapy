"""Plot background model and store as cube so that it can viewed with ds9.
"""
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table


def _make_bin_edges_array(lo, hi):
    return np.append(lo.flatten(), hi.flatten()[-1:])


class CubeBackgroundModel(object):
    """Cube background model.

    TODO: this is a prototype that can only read and plot a cube.
    Should be integrated into gammapy.background or gammapy.irf.
    """

    def __init__(self, det_bins, energy_bins, background):
        self.det_bins = det_bins
        self.energy_bins = energy_bins

        # TODO: what's the axes order?
        # ENERGY, DETX, DETY
        # or
        # ENERGY, DETY, DETX
        self.background = background

    @staticmethod
    def read(filename):

        # TODO: should we use the Table class here to read the data?
        hdu_list = fits.open(filename)
        hdu = hdu_list['BACKGROUND']
        data = hdu.data

        det_bins = _make_bin_edges_array(data['DETX_LO'], data['DETX_HI'])
        det_bins = Angle(det_bins, 'deg')
        energy_bins = _make_bin_edges_array(data['ENERG_LO'], data['ENERG_HI'])
        energy_bins = Quantity(energy_bins, 'TeV')
        background = data['Bgd'][0]

        return CubeBackgroundModel(det_bins=det_bins,
                                   energy_bins=energy_bins,
                                   background=background)

    @property
    def image_extent(self):
        """Image extent `(x_lo, x_hi, y_lo, y_hi)` in deg."""
        b = self.det_bins.degree
        return [b[0], b[-1], b[0], b[-1]]


    def plot_images(self, filename=None):
        import matplotlib.pyplot as plt

        nimages = len(self.energy_bins) - 1
        ncols = int(np.sqrt(nimages)) + 1
        nrows = (nimages // ncols) + 1
        # print(nimages, ncols, nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

        extent = self.image_extent

        for ii in range(nimages):
            data = self.background[ii]
            energy = self.energy_bins[ii]
            ax = axes.flat[ii]
            image = ax.imshow(data, extent=extent, interpolation='nearest',
                              cmap='afmhot')
            ax.set_title('Energy = {:.1f}'.format(energy))
            # fig.colorbar(image)
        # import IPython; IPython.embed()

        if filename:
            print('Wrinting {}'.format(filename))
            fig.savefig(filename)

    def plot_spectra(self, filename):
        raise NotImplementedError

    def write_cube(self, filename):
        hdu = fits.ImageHDU(data=self.background)
        print('Writing {}'.format(filename))
        hdu.writeto(filename, clobber=True)


def plot_example():
    DIR = '/Users/deil/work/_Data/hess/HESSFITS/pa/Model_Deconvoluted_Prod26/Mpp_Std/background/'
    filename = DIR + 'hist_alt3_az0.fits.gz'
    bg_model = CubeBackgroundModel.read(filename)
    bg_model.plot_images('cube_background_model.png')
    bg_model.write_cube('cube_background_model.fits')

if __name__ == '__main__':
    plot_example()
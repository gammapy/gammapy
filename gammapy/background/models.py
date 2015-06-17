# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background models.
"""
from __future__ import print_function, division
import numpy as np
from astropy.modeling.models import Gaussian1D
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table

__all__ = ['GaussianBand2D',
           'CubeBackgroundModel',
           ]

DEFAULT_SPLINE_KWARGS = dict(k=1, s=0)


class GaussianBand2D(object):
    """Gaussian band model.

    This 2-dimensional model is Gaussian in ``y`` for a given ``x``,
    and the Gaussian parameters can vary in ``x``.

    One application of this model is the diffuse emission along the
    Galactic plane, i.e. ``x = GLON`` and ``y = GLAT``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table of Gaussian parameters.
        ``x``, ``amplitude``, ``mean``, ``stddev``.
    spline_kwargs : dict
        Keyword arguments passed to `~scipy.interpolate.UnivariateSpline`
    """

    def __init__(self, table, spline_kwargs=DEFAULT_SPLINE_KWARGS):
        self.table = table
        self.parnames = ['amplitude', 'mean', 'stddev']

        from scipy.interpolate import UnivariateSpline
        s = dict()
        for parname in self.parnames:
            x = self.table['x']
            y = self.table[parname]
            s[parname] = UnivariateSpline(x, y, **spline_kwargs)
        self._par_model = s

    def _evaluate_y(self, y, pars):
        """Evaluate Gaussian model at a given ``y`` position.
        """
        return Gaussian1D.evaluate(y, **pars)

    def parvals(self, x):
        """Interpolated parameter values at a given ``x``.
        """
        x = np.asanyarray(x, dtype=float)
        parvals = dict()
        for parname in self.parnames:
            par_model = self._par_model[parname]
            shape = x.shape
            parvals[parname] = par_model(x.flat).reshape(shape)

        return parvals

    def y_model(self, x):
        """Create model at a given ``x`` position.
        """
        x = np.asanyarray(x, dtype=float)
        parvals = self.parvals(x)
        return Gaussian1D(**parvals)

    def evaluate(self, x, y):
        """Evaluate model at a given position ``(x, y)`` position.
        """
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)
        parvals = self.parvals(x)
        return self._evaluate_y(y, parvals)


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
            print('Writing {}'.format(filename))
            fig.savefig(filename)

    def plot_spectra(self, filename):
        raise NotImplementedError

    def write_cube(self, filename):
        hdu = fits.ImageHDU(data=self.background)
        print('Writing {}'.format(filename))
        hdu.writeto(filename, clobber=True)

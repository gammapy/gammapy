# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background models.
"""
from __future__ import print_function, division
import numpy as np
from astropy.modeling.models import Gaussian1D
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from ..utils.wcs import (linear_wcs_to_arrays,
                         linear_arrays_to_wcs)
from ..utils.fits import table_to_fits_table

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
    """Make bin edges array from a low values and a high values array.

    TODO: move this function to somewhere else? (i.e. utils?)

    Parameters
    ----------
    lo : `~numpy.ndarray`
        Lower boundaries.
    hi : `~numpy.ndarray`
        Higher boundaries.

    Returns
    -------
    bin_edges : `~numpy.ndarray`
        Array of bin edges as ``[[low], [high]]``.
    """
    return np.append(lo.flatten(), hi.flatten()[-1:])


def _parse_bg_units(background_unit):
    """
    Utility function to parse the bg units correctly.
    """
    # try 1st to parse them as astropy units
    try:
        u.Unit(background_unit)
    # if it fails, try to parse them manually
    except ValueError:
        tev_units = ['1/s/TeV/sr', 's-1 sr-1 TeV-1', '1 / (s sr TeV)',
                     '1 / (TeV s sr)']
        mev_units = ['1/s/MeV/sr', 'MeV-1 s-1 sr-1', '1 / (s sr MeV)',
                     '1 / (MeV s sr)']
        if background_unit in tev_units:
            background_unit = '1 / (s TeV sr)'
        elif background_unit in mev_units:
            background_unit = '1 / (s MeV sr)'
        # if it still fails, raise an exception
        else:
            raise ValueError("Cannot interpret units ({})".format(background_unit))

    return background_unit


class CubeBackgroundModel(object):

    """Cube background model.

    Container class for cube background model *(X, Y, energy)*.
    *(X, Y)* are detector coordinates (a.k.a. nominal system).
    The class hass methods for reading a model from a fits file,
    write a model to a fits file and plot the models.

    The order of the axes in the background cube is **(E, y, x)**,
    so in order to access the data correctly, the call is
    ``bg_cube_model.background[energy_bin, dety_bin, detx_bin]``.

    Parameters
    ----------
    detx_bins : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high). X coordinate.
    dety_bins : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high). Y coordinate.
    energy_bins : `~astropy.units.Quantity`
        Energy bin edges vector (low and high).
    background : `~astropy.units.Quantity`
        Background cube in (energy, X, Y) format.

    Examples
    --------
    Access cube bg model data:

    .. code:: python

        energy_bin = bg_cube_model.find_energy_bin(energy=Quantity(2., 'TeV'))
        det_bin = bg_cube_model.find_det_bin(det=Angle([0., 0.], 'degree'))
        bg_cube_model.background[energy_bin, det_bin[1], det_bin[0]]
    """

    def __init__(self, detx_bins, dety_bins, energy_bins, background):
        self.detx_bins = detx_bins
        self.dety_bins = dety_bins
        self.energy_bins = energy_bins

        self.background = background

    @classmethod
    def from_fits_table(cls, hdu):
        """Read cube background model from a fits binary table.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            HDU binary table for the bg cube

        Returns
        -------
        bg_cube : `~gammapy.background.CubeBackgroundModel`
            bg model cube object
        """

        header = hdu.header
        data = hdu.data

        # check correct axis order: 1st X, 2nd Y, 3rd energy, 4th bg
        if (header['TTYPE1'] != 'DETX_LO') or (header['TTYPE2'] != 'DETX_HI'):
            raise ValueError("Expecting X axis in first 2 places, not ({0}, {1})"
                             .format(header['TTYPE1'], header['TTYPE2']))
        if (header['TTYPE3'] != 'DETY_LO') or (header['TTYPE4'] != 'DETY_HI'):
            raise ValueError("Expecting Y axis in second 2 places, not ({0}, {1})"
                             .format(header['TTYPE3'], header['TTYPE4']))
        if (header['TTYPE5'] != 'ENERG_LO') or (header['TTYPE6'] != 'ENERG_HI'):
            raise ValueError("Expecting E axis in third 2 places, not ({0}, {1})"
                             .format(header['TTYPE5'], header['TTYPE6']))
        if (header['TTYPE7'] != 'Bgd'):
            raise ValueError("Expecting bg axis in fourth place, not ({})"
                             .format(header['TTYPE7']))

        # get det X, Y binning
        detx_bins = _make_bin_edges_array(data['DETX_LO'], data['DETX_HI'])
        dety_bins = _make_bin_edges_array(data['DETY_LO'], data['DETY_HI'])
        if header['TUNIT1'] == header['TUNIT2']:
            detx_unit = header['TUNIT1']
        else:
            raise ValueError("Detector X units not matching ({0}, {1})"
                             .format(header['TUNIT1'], header['TUNIT2']))
        if header['TUNIT3'] == header['TUNIT4']:
            dety_unit = header['TUNIT3']
        else:
            raise ValueError("Detector Y units not matching ({0}, {1})"
                             .format(header['TUNIT3'], header['TUNIT4']))
        if not detx_unit == dety_unit:
            ss_error = "This is odd: detector X and Y units not matching"
            ss_error += "({0}, {1})".format(detx_unit, dety_unit)
            raise ValueError(ss_error)
        detx_bins = Angle(detx_bins, detx_unit)
        dety_bins = Angle(dety_bins, dety_unit)

        # get energy binning
        energy_bins = _make_bin_edges_array(data['ENERG_LO'], data['ENERG_HI'])
        if header['TUNIT5'] == header['TUNIT6']:
            energy_unit = header['TUNIT5']
        else:
            raise ValueError("Energy units not matching ({0}, {1})"
                             .format(header['TUNIT5'], header['TUNIT6']))
        energy_bins = Quantity(energy_bins, energy_unit)

        # get background data
        background = data['Bgd'][0]
        background_unit = _parse_bg_units(header['TUNIT7'])
        background = Quantity(background, background_unit)

        return cls(detx_bins=detx_bins,
                   dety_bins=dety_bins,
                   energy_bins=energy_bins,
                   background=background)

    @classmethod
    def from_fits_image(cls, image_hdu, energy_hdu):
        """Read cube background model from a fits image.

        Parameters
        ----------
        image_hdu : `~astropy.io.fits.PrimaryHDU`
            Background cube image HDU
        energy_hdu : `~astropy.io.fits.BinTableHDU`
            Energy binning table

        Returns
        -------
        bg_cube : `~gammapy.background.CubeBackgroundModel`
            Background cube
        """
        image_header = image_hdu.header
        energy_header = energy_hdu.header

        # check correct axis order: 1st X, 2nd Y, 3rd energy, 4th bg
        if (image_header['CTYPE1'] != 'DETX'):
            raise ValueError("Expecting X axis in first place, not ({})"
                             .format(image_header['CTYPE1']))
        if (image_header['CTYPE2'] != 'DETY'):
            raise ValueError("Expecting Y axis in second place, not ({})"
                             .format(image_header['CTYPE2']))
        if (image_header['CTYPE3'] != 'ENERGY'):
            raise ValueError("Expecting E axis in third place, not ({})"
                             .format(image_header['CTYPE3']))

        # check units
        if (image_header['CUNIT1'] != image_header['CUNIT2']):
            ss_error = "This is odd: detector X and Y units not matching"
            ss_error += "({0}, {1})".format(image_header['CUNIT1'], image_header['CUNIT2'])
            raise ValueError(ss_error)
        if (image_header['CUNIT3'] != energy_header['TUNIT1']):
            ss_error = "This is odd: energy units not matching"
            ss_error += "({0}, {1})".format(image_header['CUNIT3'], energy_header['TUNIT1'])
            raise ValueError(ss_error)

        # get det X, Y binning
        wcs = WCS(image_header, naxis=2) # select only the (X, Y) axes
        detx_bins, dety_bins = linear_wcs_to_arrays(wcs,
                                                    image_header['NAXIS1'],
                                                    image_header['NAXIS2'])

        # get energy binning
        energy_bins = Quantity(energy_hdu.data['ENERGY'],
                               energy_header['TUNIT1'])

        # get background data
        background = image_hdu.data
        background_unit = _parse_bg_units(image_header['BG_UNIT'])
        background = Quantity(background, background_unit)

        return cls(detx_bins=detx_bins,
                   dety_bins=dety_bins,
                   energy_bins=energy_bins,
                   background=background)

    @classmethod
    def read(cls, filename, format='table'):
        """Read cube background model from fits file.

        Several input formats are accepted, depending on the value
        of the **format** parameter:

        * table (default and preferred format): `~astropy.io.fits.BinTableHDU`

        * image (alternative format): `~astropy.io.fits.PrimaryHDU`,
          with the energy binning stored as `~astropy.io.fits.BinTableHDU`

        Parameters
        ----------
        filename : str
            name of file with the bg cube
        format : str, optional
            format of the bg cube to read

        Returns
        -------
        bg_cube : `~gammapy.background.CubeBackgroundModel`
            bg model cube object
        """
        hdu = fits.open(filename)
        if format == 'table':
            return cls.from_fits_table(hdu['BACKGROUND'])
        elif format == 'image':
            return cls.from_fits_image(hdu['PRIMARY'], hdu['EBOUNDS'])
        else:
            raise ValueError("Invalid format {}.".format(format))

    def to_table(self):
        """Convert cube background model to astropy table format.

        The name of the table is stored in the table meta information
        under the keyword 'name'.

        Returns
        -------
        table : `~astropy.table.Table`
            table containing the bg cube
        """
        # data arrays
        a_detx_lo = Quantity([self.detx_bins[:-1]])
        a_detx_hi = Quantity([self.detx_bins[1:]])
        a_dety_lo = Quantity([self.dety_bins[:-1]])
        a_dety_hi = Quantity([self.dety_bins[1:]])
        a_energy_lo = Quantity([self.energy_bins[:-1]])
        a_energy_hi = Quantity([self.energy_bins[1:]])
        a_bg = Quantity([self.background])

        # table
        table = Table()
        table['DETX_LO'] = a_detx_lo
        table['DETX_HI'] = a_detx_hi
        table['DETY_LO'] = a_dety_lo
        table['DETY_HI'] = a_dety_hi
        table['ENERG_LO'] = a_energy_lo
        table['ENERG_HI'] = a_energy_hi
        table['Bgd'] = a_bg

        table.meta['name'] = 'BACKGROUND'

        return table

    def to_fits_table(self):
        """Convert cube background model to binary table fits format.

        Returns
        -------
        tbhdu : `~astropy.io.fits.BinTableHDU`
            table containing the bg cube
        """
        return table_to_fits_table(self.to_table())

    def to_fits_image(self):
        """Convert cube background model to image fits format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with:

            * one `~astropy.io.fits.PrimaryHDU` image for the bg cube

            * one `~astropy.io.fits.BinTableHDU` table for the energy binning
        """
        # data
        imhdu = fits.PrimaryHDU(data=self.background.value,
                                header=self.det_wcs.to_header())
        # add some important header information
        imhdu.header['BG_UNIT'] = '{0.unit:FITS}'.format(self.background)
        imhdu.header['CTYPE3'] = 'ENERGY'
        imhdu.header['CUNIT3'] = '{0.unit:FITS}'.format(self.energy_bins)

        # get WCS object and write it out as a FITS header
        wcs_header = self.det_wcs.to_header()

        # get energy values as a table HDU, via an astropy table
        energy_table = Table()
        energy_table['ENERGY'] = self.energy_bins
        energy_table.meta['name'] = 'EBOUNDS'
        # TODO: this function should be reviewed/re-written, when
        # the following PR is completed:
        # https://github.com/gammapy/gammapy/pull/290
        # as suggested in:
        # https://github.com/gammapy/gammapy/pull/299#discussion_r35044977

        enhdu = table_to_fits_table(energy_table)

        hdu_list = fits.HDUList([imhdu, enhdu])

        return hdu_list

    def write(self, outfile, format='table', **kwargs):
        """Write cube background model to fits file.

        Several output formats are accepted, depending on the value
        of the **format** parameter:

        * table (default and preferred format): `~astropy.io.fits.BinTableHDU`
        * image (alternative format): `~astropy.io.fits.PrimaryHDU`,
          with the energy binning stored as `~astropy.io.fits.BinTableHDU`

        Depending on the value of the **format** parameter, this
        method calls either `~astropy.io.fits.BinTableHDU.writeto` or
        `~astropy.io.fits.HDUList.writeto`, forwarding the
        **kwargs** arguments.

        Parameters
        ----------
        outfile : str
            name of file to write
        format : str, optional
            format of the bg cube to write
        kwargs
            extra arguments for the corresponding `io.fits` `writeto` method
        """
        if format == 'table':
            self.to_fits_table().writeto(outfile, **kwargs)
        elif format == 'image':
            self.to_fits_image().writeto(outfile, **kwargs)
        else:
            raise ValueError("Invalid format {}.".format(format))

    @property
    def image_extent(self):
        """Image extent (`~astropy.coordinates.Angle`)

        The output array format is ``(x_lo, x_hi, y_lo, y_hi)``.
        """
        bx = self.detx_bins
        by = self.dety_bins
        return Angle([bx[0], bx[-1], by[0], by[-1]])

    @property
    def spectrum_extent(self):
        """Spectrum extent (`~astropy.units.Quantity`)

        The output array format is  ``(e_lo, e_hi)``.
        """
        b = self.energy_bins
        return Quantity([b[0], b[-1]])

    @property
    def image_bin_centers(self):
        """Image bin centers **(x, y)** (2x `~astropy.coordinates.Angle`)

        Returning two separate elements for the X and Y bin centers.
        """
        detx_bin_centers = 0.5 * (self.detx_bins[:-1] + self.detx_bins[1:])
        dety_bin_centers = 0.5 * (self.dety_bins[:-1] + self.dety_bins[1:])
        return detx_bin_centers, dety_bin_centers

    @property
    def energy_bin_centers(self):
        """Energy bin centers (logarithmic center) (`~astropy.units.Quantity`)"""
        log_bin_edges = np.log(self.energy_bins.value)
        log_bin_centers = 0.5 * (log_bin_edges[:-1] + log_bin_edges[1:])
        energy_bin_centers = Quantity(np.exp(log_bin_centers), self.energy_bins.unit)
        # TODO: this function should be reviewed/re-written, when
        # the following PR is completed:
        # https://github.com/gammapy/gammapy/pull/290
        # as suggested in:
        # https://github.com/gammapy/gammapy/pull/292#discussion_r34412865
        return energy_bin_centers

    @property
    def det_wcs(self):
        """WCS object describing the coordinates of the det (X, Y) bins (`~astropy.wcs.WCS`)

        This method gives the correct answer only for linear X, Y binning.
        """
        wcs = linear_arrays_to_wcs(name_x="DETX",
                                   name_y="DETY",
                                   bin_edges_x=self.detx_bins,
                                   bin_edges_y=self.detx_bins)
        return wcs

    def find_det_bin(self, det):
        """Find the bins that contain the specified det (X, Y) pairs.

        Parameters
        ----------
        det : `~astropy.coordinates.Angle`
            array of det (X, Y) pairs to search for

        Returns
        -------
        bin_index : `~numpy.ndarray`
            array of integers with the indices (x, y) of the det
            bin containing the specified det (X, Y) pair
        """
        # check that the specified det is within the boundaries of the model
        det_extent = self.image_extent
        check_x_lo = (det_extent[0] <= det[0]).all()
        check_x_hi = (det[0] < det_extent[1]).all()
        check_y_lo = (det_extent[2] <= det[1]).all()
        check_y_hi = (det[1] < det_extent[3]).all()
        if not (check_x_lo and check_x_hi) or not (check_y_lo and check_y_hi):
            raise ValueError("Specified det {0} is outside the boundaries {1}."
                             .format(det, det_extent))

        bin_index_x = np.searchsorted(self.detx_bins[1:], det[0])
        bin_index_y = np.searchsorted(self.dety_bins[1:], det[1])

        return np.array([bin_index_x, bin_index_y])

    def find_energy_bin(self, energy):
        """Find the bins that contain the specified energy values.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            array of energies to search for

        Returns
        -------
        bin_index : `~numpy.ndarray`
            indices of the energy bins containing the specified energies
        """
        # check that the specified energy is within the boundaries of the model
        energy_extent = self.spectrum_extent
        if not (energy_extent[0] <= energy).all() and (energy < energy_extent[1]).all():
            ss_error = "Specified energy {}".format(energy)
            ss_error += " is outside the boundaries {}.".format(energy_extent)
            raise ValueError(ss_error)

        bin_index = np.searchsorted(self.energy_bins[1:], energy)

        return bin_index

    def find_det_bin_edges(self, det):
        """Find the bin edges of the specified det (X, Y) pairs.

        Parameters
        ----------
        det : `~astropy.coordinates.Angle`
            array of det (X, Y) pairs to search for

        Returns
        -------
        bin_edges : `~astropy.coordinates.Angle`
            det bin edges (x_lo, x_hi, y_lo, y_hi)
        """
        bin_index = self.find_det_bin(det)
        bin_edges = Angle([self.detx_bins[bin_index[0]],
                           self.detx_bins[bin_index[0] + 1],
                           self.dety_bins[bin_index[1]],
                           self.dety_bins[bin_index[1] + 1]])

        return bin_edges

    def find_energy_bin_edges(self, energy):
        """Find the bin edges of the specified energy values.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            array of energies to search for

        Returns
        -------
        bin_edges : `~astropy.units.Quantity`
            energy bin edges [E_min, E_max)
        """
        bin_index = self.find_energy_bin(energy)
        bin_edges = Quantity([self.energy_bins[bin_index],
                              self.energy_bins[bin_index + 1]])

        return bin_edges

    def plot_image(self, energy, ax=None, style_kwargs=None):
        """Plot image for the energy bin containing the specified energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy of bin to plot the bg model
        ax : `~matplotlib.axes.Axes`, optional
            axes of the figure for the plot
        style_kwargs : dict, optional
            style options for the plot

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            axes of the figure containing the plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        energy = energy.flatten()
        # check shape of energy: only 1 value is accepted
        nvalues = len(energy)
        if nvalues != 1:
            ss_error = "Expected exactly 1 value for energy, got {}.".format(nvalues)
            raise IndexError(ss_error)
        else:
            energy = Quantity(energy[0])

        extent = self.image_extent
        energy_bin_centers = self.energy_bin_centers

        # find energy bin containing the specified energy
        energy_bin = self.find_energy_bin(energy)
        energy_bin_edges = self.find_energy_bin_edges(energy)
        ss_energy_bin_edges = "[{0}, {1}) {2}".format(energy_bin_edges[0].value,
                                                      energy_bin_edges[1].value,
                                                      energy_bin_edges.unit)

        # get data for the plot
        data = self.background[energy_bin]
        energy_bin_center = energy_bin_centers[energy_bin]

        # create plot
        fig = plt.figure()
        do_not_close_fig = False
        if ax is None:
            ax = fig.add_subplot(111)
            # if no axis object is passed by ref, the figure should remain open
            do_not_close_fig = True
        if style_kwargs is None:
            style_kwargs = dict()

        fig.set_size_inches(8., 8., forward=True)
        #import IPython; IPython.embed()

        if not 'cmap' in style_kwargs:
            style_kwargs['cmap'] = 'afmhot'

        image = ax.imshow(data.value,
                          extent=extent.value,
                          origin='lower', # do not invert image
                          interpolation='nearest',
                          **style_kwargs)

        # set title and axis names
        ax.set_title('Energy = [{0:.1f}, {1:.1f}) {2}'.format(energy_bin_edges[0].value,
                                                              energy_bin_edges[1].value,
                                                              energy_bin_edges.unit))
        ax.set_xlabel('X / {}'.format(extent.unit))
        ax.set_ylabel('Y / {}'.format(extent.unit))

        # draw color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(image, cax=cax, label='Bg rate / {}'.format(data.unit))

        # eventually close figure to avoid white canvases
        if not do_not_close_fig:
            plt.close(fig)
        return ax

    def plot_spectrum(self, det, ax=None, style_kwargs=None):
        """Plot spectra for the det bin containing the specified det (X, Y) pair.

        Parameters
        ----------
        det : `~astropy.units.Quantity`
            det (X,Y) pair of bin to plot the bg model
        ax : `~matplotlib.axes.Axes`, optional
            axes of the figure for the plot
        style_kwargs : dict, optional
            style options for the plot

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            axes of the figure containing the plot
        """
        import matplotlib.pyplot as plt

        det = det.flatten() # flatten
        # check shape of det: only 1 pair is accepted
        nvalues = len(det.flatten())
        if nvalues != 2:
            ss_error = "Expected exactly 2 values for det (X, Y),"
            ss_error += "got {}.".format(nvalues)
            raise IndexError(ss_error)
        else:
            do_only_1_plot = True

        energy_points = self.energy_bin_centers
        detx_bin_centers, dety_bin_centers = self.image_bin_centers

        # find det bin containing the specified det coordinates
        det_bin = self.find_det_bin(det)
        det_bin_edges = self.find_det_bin_edges(det)
        ss_detx_bin_edges = "[{0}, {1}) {2}".format(det_bin_edges[0].value,
                                                    det_bin_edges[1].value,
                                                    det_bin_edges.unit)
        ss_dety_bin_edges = "[{0}, {1}) {2}".format(det_bin_edges[2].value,
                                                    det_bin_edges[3].value,
                                                    det_bin_edges.unit)

        # get data for the plot
        data = self.background[:, det_bin[1], det_bin[0]]
        detx_bin_center = detx_bin_centers[det_bin[0]]
        dety_bin_center = dety_bin_centers[det_bin[1]]

        # create plot
        fig = plt.figure()
        do_not_close_fig = False
        if ax is None:
            ax = fig.add_subplot(111)
            # if no axis object is passed by ref, the figure should remain open
            do_not_close_fig = True
        if style_kwargs is None:
            style_kwargs = dict()

        fig.set_size_inches(8., 8., forward=True)

        image = ax.plot(energy_points.to('TeV'),
                        data,
                        drawstyle='default', # connect points with lines
                        **style_kwargs)
        ax.loglog() # double log scale # slow!

        # set title and axis names
        ss_detx_bin_edges = "[{0:.1f}, {1:.1f}) {2}".format(det_bin_edges[0].value,
                                                            det_bin_edges[1].value,
                                                            det_bin_edges.unit)
        ss_dety_bin_edges = "[{0:.1f}, {1:.1f}) {2}".format(det_bin_edges[2].value,
                                                            det_bin_edges[3].value,
                                                            det_bin_edges.unit)

        ax.set_title('Det = {0} {1}'.format(ss_detx_bin_edges, ss_dety_bin_edges))
        ax.set_xlabel('E / {}'.format(energy_points.unit))
        ax.set_ylabel('Bg rate / {}'.format(data.unit))

        # eventually close figure to avoid white canvases
        if not do_not_close_fig:
            plt.close(fig)
        return ax

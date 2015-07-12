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
from astropy import wcs
from ..utils.wcs import (linear_arrays_from_wcs,
                         linear_wcs_from_arrays)

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
    lo : `~numpy.array`
        lower boundaries
    hi : `~numpy.array`
        higher boundaries

    Returns
    -------
    bin_edges : `~numpy.array`
        array of bin edges as [[low], [high]]
    """
    return np.append(lo.flatten(), hi.flatten()[-1:])


class CubeBackgroundModel(object):

    """Cube background model.

    Container class for cube background model (X, Y, energy).
    (X, Y) are detector coordinates (a.k.a. nominal system).
    The class hass methods for reading a model from a fits file,
    write a model to a fits file and plot the models.

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
    """

    def __init__(self, detx_bins, dety_bins, energy_bins, background):
        self.detx_bins = detx_bins
        self.dety_bins = dety_bins
        self.energy_bins = energy_bins

        self.background = background

    @staticmethod
    def from_fits_bin_table(tbhdu):
        """Read cube background model from a fits binary table.

        Parameters
        ----------
        tbhdu : `~astropy.io.fits.BinTableHDU`
            HDU binary table for the bg cube

        Returns
        -------
        bg_cube : `~gammapy.models.CubeBackgroundModel`
            bg model cube object
        """
 
        header = tbhdu.header
        data = tbhdu.data

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
        background_unit = header['TUNIT7']
        tev_units = ['1/s/TeV/sr', 's-1 sr-1 TeV-1', '1 / (s sr TeV)',
                     '1 / (TeV s sr)']
        mev_units = ['1/s/MeV/sr', 'MeV-1 s-1 sr-1', '1 / (s sr MeV)',
                     '1 / (MeV s sr)']
        if background_unit in tev_units:
            background_unit = '1 / (s TeV sr)'
        elif background_unit in mev_units:
            background_unit = '1 / (s MeV sr)'
        else:
            raise ValueError("Cannot interpret units ({})".format(background_unit))
        background = Quantity(background, background_unit)

        return CubeBackgroundModel(detx_bins=detx_bins,
                                   dety_bins=dety_bins,
                                   energy_bins=energy_bins,
                                   background=background)

    @staticmethod
    def from_fits_image(imhdu):
        """Read cube background model from a fits image.

        Parameters
        ----------
        imhdu : `~astropy.io.fits.ImageHDU`
            HDU image for the bg cube

        Returns
        -------
        bg_cube : `~gammapy.models.CubeBackgroundModel`
            bg model cube object
        """
        raise NotImplementedError

    @staticmethod
    def read(filename, format='table'):
        """Read cube background model from fits file.

        Several input formats are accepted, depending on the value
        of the 'format' parameter:

        * bin_table (default and preferred format): `~astropy.io.fits.BinTableHDU`
        * image (alternative format): `~astropy.io.fits.ImageHDU`

        Parameters
        ----------
        filename : `~str`
            name of file with the bg cube
        format : `~str`, optional
            format of the bg cube to read

        Returns
        -------
        bg_cube : `~gammapy.models.CubeBackgroundModel`
            bg model cube object
        """
        hdu = fits.open(filename)
        hdu = hdu['BACKGROUND']
        if format == 'table':
            return CubeBackgroundModel.from_fits_bin_table(hdu)
        elif format == 'image':
            return CubeBackgroundModel.from_fits_image(hdu)
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

        table.meta['E_THRES'] = a_energy_lo.flatten()[0].value

        # name
        table.meta['name'] = 'BACKGROUND'

        return table

    def to_fits_bin_table(self):
        """Convert cube background model to binary table fits format.

        Returns
        -------
        tbhdu : `~astropy.io.fits.BinTableHDU`
            table containing the bg cube
        """
        # build astropy table
        table = self.to_table()

        # read name and drop it from the meta information, otherwise
        # it would be stored as a header keyword in the BinTableHDU
        name = table.meta.popitem('name')[1]

        data = table.as_array()

        header = fits.Header()
        header.update(table.meta)

        tbhdu = fits.BinTableHDU(data, header, name=name)

        # Copy over column meta-data
        for colname in table.colnames:
            tbhdu.columns[colname].unit = str(table[colname].unit)

        # TODO: this method works fine but the order of keywords in the table
        # header is not logical: for instance, list of keywords with column
        # units (TUNITi) is appended after the list of column keywords
        # (TTYPEi, TFORMi), instead of in between.
        # As a matter of fact, the units aren't yet in the header, but
        # only when calling the write method and opening the output file.
        # https://github.com/gammapy/gammapy/issues/298

        return tbhdu

    def to_fits_image(self):
        """Convert cube background model to image fits format.

        Returns
        -------
        imhdu : `~astropy.io.fits.ImageHDU`
            image containing the bg cube
        """
        imhdu = fits.ImageHDU(data=self.background.value, name='BACKGROUND')
        # TODO: store units (of bg) somewhere in header??!!!!
        # TODO: implement WCS object to be able to read the det coords -> done
        # TODO: energy binning: store in HDU table like for SpectralCube class

        # get WCS object and write it out as a FITS header
        wcs_header = self.det_wcs.to_header()

        # transferring header values
        imhdu.header.update(wcs_header)

        return imhdu

    def write(self, outfile, format='table', write_kwargs=None):
        """Write cube background model to fits file.

        Several output formats are accepted, depending on the value
        of the `format` parameter:

        * bin_table (default and preferred format): `~astropy.io.fits.BinTableHDU`
        * image (alternative format): `~astropy.io.fits.ImageHDU`

        Depending on the value of the `format` parameter, this
        method calls either `~astropy.io.fits.BinTableHDU.writeto` or
        `~astropy.io.fits.ImageHDU.writeto`, forwarding the
        `write_kwargs` arguments.

        Parameters
        ----------
        outfile : `~str`
            name of file to write
        format : `~str`, optional
            format of the bg cube to write
        write_kwargs : `~dict`, optional
            extra arguments for the corresponding `io.fits` `writeto` method
        """
        if write_kwargs is None:
            write_kwargs = dict()

        if format == 'table':
            self.to_fits_bin_table().writeto(outfile, **write_kwargs)
        elif format == 'image':
            self.to_fits_image().writeto(outfile, **write_kwargs)
        else:
            raise ValueError("Invalid format {}.".format(format))

    @property
    def image_extent(self):
        """Image extent `(x_lo, x_hi, y_lo, y_hi)`.

        Returns
        -------
        im_extent : `~astropy.coordinates.Angle`
            array of bins with the image extent
        """
        bx = self.detx_bins
        by = self.dety_bins
        return Angle([bx[0], bx[-1], by[0], by[-1]])

    @property
    def spectrum_extent(self):
        """Spectrum extent `(e_lo, e_hi)`.

        Returns
        -------
        spec_extent : `~astropy.units.Quantity`
            array of bins with the spectrum extent
        """
        b = self.energy_bins
        return Quantity([b[0], b[-1]])

    @property
    def image_bin_centers(self):
        """Image bin centers `(x, y)`.

        Returns
        -------
        det_edges_centers : `~astropy.coordinates.Angle`
            array of bins with the image bin centers [[x_centers], [y_centers]]
        """
        detx_edges_low = self.detx_bins[:-1]
        detx_edges_high = self.detx_bins[1:]
        detx_edges_centers = (detx_edges_low + detx_edges_high)/2.
        dety_edges_low = self.dety_bins[:-1]
        dety_edges_high = self.dety_bins[1:]
        dety_edges_centers = (dety_edges_low + dety_edges_high)/2.
        return Angle([detx_edges_centers, dety_edges_centers])

    @property
    def energy_bin_centers(self):
        """Energy bin centers (logarithmic center).

        Returns
        -------
        energy_bin_centers : `~astropy.units.Quantity`
            array of bins with the spectrum bin centers
        """
        energy_edges_low = self.energy_bins[:-1]
        energy_edges_high = self.energy_bins[1:]
        e_lo_tev = energy_edges_low.to('TeV').value
        e_hi_tev = energy_edges_high.to('TeV').value
        energy_bin_centers = 10.**(0.5*(np.log10(e_lo_tev*e_hi_tev)))
        # TODO: this function should be reviewed/re-written, when
        # the following PR is completed:
        # https://github.com/gammapy/gammapy/pull/290
        # as suggested in:
        # https://github.com/gammapy/gammapy/pull/292#discussion_r34412865
        return Quantity(energy_bin_centers, 'TeV')

    @property
    def det_wcs(self):
        """WCS object describing the coordinates of the det (X, Y) bins.

        This method gives the correct answer only for linear X, Y binning.

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            WCS object describing the bin coordinates
        """
        wcs = linear_wcs_from_arrays(name_x="DETX",
                                     name_y="DETY",
                                     bins_x=self.detx_bins,
                                     bins_y=self.detx_bins)
        return wcs

    def find_det_bin(self, det):
        """Find the bin that contains the specified det (X, Y) pair.

        TODO: implement test as suggested in:
            https://github.com/gammapy/gammapy/pull/292#discussion_r33843508

        Parameters
        ----------
        det : `~astropy.coordinates.Angle`
            det (X, Y) pair to search for

        Returns
        -------
        bin_pos : `~int`
            index of the det bin containing the specified det (X, Y) pair
        bin_edges : `~astropy.units.Quantity`
            det bin edges (x_lo, x_hi, y_lo, y_hi)
        """
        # check shape of det: only 1 pair is accepted
        nvalues = len(det.flatten())
        if nvalues != 2:
            raise IndexError("Expected exactly 2 values for det (X, Y), got {}."
                             .format(nvalues))

        # check that the specified det is within the boundaries of the model
        det_extent = self.image_extent
        check_x_lo = (det_extent[0] <= det[0])
        check_x_hi = (det[0] < det_extent[1])
        check_y_lo = (det_extent[2] <= det[1])
        check_y_hi = (det[1] < det_extent[3])
        if not (check_x_lo and check_x_hi) or not (check_y_lo and check_y_hi):
            raise ValueError("Specified det {0} is outside the boundaries {1}."
                             .format(det, det_extent))

        detx_edges_low = self.detx_bins[:-1]
        detx_edges_high = self.detx_bins[1:]
        dety_edges_low = self.dety_bins[:-1]
        dety_edges_high = self.dety_bins[1:]
        bin_pos_x = np.searchsorted(detx_edges_high, det[0])
        bin_pos_y = np.searchsorted(dety_edges_high, det[1])
        bin_pos = np.array([bin_pos_x, bin_pos_y])
        bin_edges = Angle([detx_edges_low[bin_pos[0]], detx_edges_high[bin_pos[0]],
                           dety_edges_low[bin_pos[1]], dety_edges_high[bin_pos[1]]])

        return bin_pos, bin_edges

    def find_energy_bin(self, energy):
        """Find the bin that contains the specified energy value.

        TODO: implement test as suggested in:
            https://github.com/gammapy/gammapy/pull/292#discussion_r33843508

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy to search for

        Returns
        -------
        bin_pos : `~int`
            index of the energy bin containing the specified energy
        bin_edges : `~astropy.units.Quantity`
            energy bin edges [E_min, E_max)
        """
        # check shape of energy: only 1 value is accepted
        nvalues = len(energy.flatten())
        if nvalues != 1:
            raise IndexError("Expected exactly 1 value for energy, got {}."
                             .format(nvalues))

        # check that the specified energy is within the boundaries of the model
        energy_extent = self.spectrum_extent
        if not (energy_extent[0] <= energy) and (energy < energy_extent[1]):
            ss_error = "Specified energy {}".format(energy)
            ss_error += " is outside the boundaries {}.".format(energy_extent)
            raise ValueError(ss_error)

        energy_edges_low = self.energy_bins[:-1]
        energy_edges_high = self.energy_bins[1:]
        bin_pos = np.searchsorted(energy_edges_high, energy)
        bin_edges = Quantity([energy_edges_low[bin_pos], energy_edges_high[bin_pos]])

        return bin_pos, bin_edges

    def plot_image(self, energy, ax=None):
        """Plot image for the energy bin containing the specified energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy of bin to plot the bg model
        ax : `~matplotlib.axes.Axes`, optional
            axes of the figure for the plot

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            axes of the figure containing the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        energy = energy.flatten() # flatten
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
        energy_bin, energy_bin_edges = self.find_energy_bin(energy)
        ss_energy_bin_edges = "[{0}, {1}) {2}".format(energy_bin_edges[0].value,
                                                      energy_bin_edges[1].value,
                                                      energy_bin_edges.unit)

        # get data for the plot
        ii = energy_bin
        data = self.background[ii]
        energy_bin_center = energy_bin_centers[ii]

        # create plot
        fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)

        fig.set_size_inches(8., 8., forward=True)

        image = ax.imshow(data.value,
                          extent=extent.value,
                          origin='lower', # do not invert image
                          interpolation='nearest',
                          norm=LogNorm(), # color log scale
                          cmap='afmhot')

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

        return ax

    def plot_spectrum(self, det, ax=None):
        """Plot spectra for the det bin containing the specified det (X, Y) pair.

        Parameters
        ----------
        det : `~astropy.units.Quantity`
            det (X,Y) pair of bin to plot the bg model
        ax : `~matplotlib.axes.Axes`, optional
            axes of the figure for the plot

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
        det_bin_centers = self.image_bin_centers

        # find det bin containing the specified det coordinates
        det_bin, det_bin_edges = self.find_det_bin(det)
        ss_detx_bin_edges = "[{0}, {1}) {2}".format(det_bin_edges[0].value,
                                                    det_bin_edges[1].value,
                                                    det_bin_edges.unit)
        ss_dety_bin_edges = "[{0}, {1}) {2}".format(det_bin_edges[2].value,
                                                    det_bin_edges[3].value,
                                                    det_bin_edges.unit)

        # get data for the plot
        ii = det_bin[0]
        jj = det_bin[1]
        data = self.background[:, ii, jj]
        detx_bin_center = det_bin_centers[0, ii]
        dety_bin_center = det_bin_centers[1, jj]

        # create plot
        fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)

        fig.set_size_inches(8., 8., forward=True)

        image = ax.plot(energy_points.to('TeV'), data,
                        drawstyle='default') # connect points with lines
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

        return ax

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Cube container.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from ..utils.wcs import (linear_wcs_to_arrays,
                         linear_arrays_to_wcs)
from ..utils.fits import table_to_fits_table

__all__ = ['Cube',
           ]


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


class Cube(object):

    """Cube container class.

    Container class for cubes *(X, Y, energy)*.
    *(X, Y)* are detector coordinates (a.k.a. nominal system).
    The class has methods for reading a cube from a fits file,
    write a cube to a fits file and plot the cubes among others.
    TODO: update this par!!!!

    The order of the axes in the cube is **(E, y, x)**,
    so in order to access the data correctly, the call is
    ``cube.data[energy_bin, dety_bin, detx_bin]``.

    - TODO: rename detx/dety to x/y!!!
    - TODO: review this doc!!!
    - TODO: review this class!!!
    - TODO: review high-level doc!!!
    - TODO: what should I do with the bg units parser???!!! (and read/write methods)!!!
    - TODO: revise imports of all files (also TEST files) at the end
    - TODO: is this class general enough to use it for other things
    - besides bg cube models? (i.e. for projected bg cube models or
    - spectral cubes?)
    - TODO: read/write from/to file might need an optional argument
    - to specify the name of the HDU
    - TODO: update also (in datasets) make_test_bg_cube_model and test_make_test_bg_cube_model !!!
    - TODO: think about the naming/definition/purpose of make_test_bg_cube_model (and the test file in gammapy-extra) !!!
    - probably I want to extend it to use the new bg cube model format (3 cubes) or, just leave it as is (and redefine naming/docs)??!!!
    - TODO: review background: make.py models.py
    - TODO: review background/tests: test_models test_cube
    - TODO: revise imports of all files (also TEST files) at the end

    list of (to) mod files:

    - gammapy/background/__init__.py
    - gammapy/background/make.py
    - gammapy/background/models.py
    - gammapy/background/tests/test_models.py
    - gammapy/datasets/make.py
    - gammapy/datasets/tests/test_make.py
    - gammapy/background/cube.py
    - gammapy/background/tests/test_cube.py
    - docs/background/plot_bgcube.py
    - examples/plot_background_model.py -> plot_bg_cube_model.py (update ref in docs!)
    - docs/background/models.rst
    - docs/background/make_models.rst

    Parameters
    ----------
    detx_bins : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high). X coordinate.
    dety_bins : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high). Y coordinate.
    energy_bins : `~astropy.units.Quantity`
        Energy bin edges vector (low and high).
    data : `~astropy.units.Quantity`
        Data cube matrix in (energy, X, Y) format.

    Examples
    --------
    Access cube data:

    .. code:: python

        energy_bin = cube.find_energy_bin(energy=Quantity(2., 'TeV'))
        det_bin = cube.find_det_bin(det=Angle([0., 0.], 'degree'))
        cube.data[energy_bin, det_bin[1], det_bin[0]]
    """

    def __init__(self, detx_bins=None, dety_bins=None, energy_bins=None, data=None):
        self.detx_bins = detx_bins
        self.dety_bins = dety_bins
        self.energy_bins = energy_bins

        self.data = data

    @classmethod
    def from_fits_table(cls, hdu):
        """Read cube from a fits binary table.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            HDU binary table for the cube.

        Returns
        -------
        cube : `~gammapy.background.Cube`
            Cube object.
        """

        header = hdu.header
        data = hdu.data

        # check correct axis order: 1st X, 2nd Y, 3rd energy, 4th data
        if (header['TTYPE1'] != 'DETX_LO') or (header['TTYPE2'] != 'DETX_HI'):
            raise ValueError("Expecting X axis in first 2 places, not ({0}, {1})"
                             .format(header['TTYPE1'], header['TTYPE2']))
        if (header['TTYPE3'] != 'DETY_LO') or (header['TTYPE4'] != 'DETY_HI'):
            raise ValueError("Expecting Y axis in second 2 places, not ({0}, {1})"
                             .format(header['TTYPE3'], header['TTYPE4']))
        if (header['TTYPE5'] != 'ENERG_LO') or (header['TTYPE6'] != 'ENERG_HI'):
            raise ValueError("Expecting E axis in third 2 places, not ({0}, {1})"
                             .format(header['TTYPE5'], header['TTYPE6']))
        if (header['TTYPE7'] != 'Bgd'): #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)
            raise ValueError("Expecting data axis in fourth place, not ({})" #TODO: show the type of the data expected (i.e. 'name' variable)!!!
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

        # get data
        data = data['Bgd'][0] #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)
        data_unit = _parse_bg_units(header['TUNIT7']) # TODO: avoid using the parser in this class!!! -> move to new BacgroundCubeModel class!!! (if necessary at all)
        data = Quantity(data, data_unit)

        return cls(detx_bins=detx_bins,
                   dety_bins=dety_bins,
                   energy_bins=energy_bins,
                   data=data)

    @classmethod
    def from_fits_image(cls, image_hdu, energy_hdu):
        """Read cube from a fits image.

        Parameters
        ----------
        image_hdu : `~astropy.io.fits.PrimaryHDU`
            Cube image HDU.
        energy_hdu : `~astropy.io.fits.BinTableHDU`
            Energy binning table.

        Returns
        -------
        cube : `~gammapy.background.Cube`
            Cube object.
        """
        image_header = image_hdu.header
        energy_header = energy_hdu.header

        # check correct axis order: 1st X, 2nd Y, 3rd energy, 4th data
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

        # get data
        data = image_hdu.data
        data_unit = _parse_bg_units(image_header['BG_UNIT']) # TODO: avoid using the parser in this class!!! -> move to new BacgroundCubeModel class!!! (if necessary at all)
        data = Quantity(data, data_unit)

        return cls(detx_bins=detx_bins,
                   dety_bins=dety_bins,
                   energy_bins=energy_bins,
                   data=data)

    @classmethod
    def read(cls, filename, format='table'):
        """Read cube from fits file.

        Several input formats are accepted, depending on the value
        of the **format** parameter:

        * table (default and preferred format): `~astropy.io.fits.BinTableHDU`

        * image (alternative format): `~astropy.io.fits.PrimaryHDU`,
          with the energy binning stored as `~astropy.io.fits.BinTableHDU`

        Parameters
        ----------
        filename : str
            Name of file with the cube.
        format : str, optional
            Format of the cube to read.

        Returns
        -------
        cube : `~gammapy.background.Cube`
            Cube object.
        """
        hdu = fits.open(filename)
        if format == 'table':
            return cls.from_fits_table(hdu['BACKGROUND']) #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)
        elif format == 'image':
            return cls.from_fits_image(hdu['PRIMARY'], hdu['EBOUNDS'])
        else:
            raise ValueError("Invalid format {}.".format(format))

    def to_table(self):
        """Convert cube to astropy table format.

        The name of the table is stored in the table meta information
        under the keyword 'name'.

        Returns
        -------
        table : `~astropy.table.Table`
            Table containing the cube.
        """
        # data arrays
        a_detx_lo = Quantity([self.detx_bins[:-1]])
        a_detx_hi = Quantity([self.detx_bins[1:]])
        a_dety_lo = Quantity([self.dety_bins[:-1]])
        a_dety_hi = Quantity([self.dety_bins[1:]])
        a_energy_lo = Quantity([self.energy_bins[:-1]])
        a_energy_hi = Quantity([self.energy_bins[1:]])
        a_data = Quantity([self.data])

        # table
        table = Table()
        table['DETX_LO'] = a_detx_lo
        table['DETX_HI'] = a_detx_hi
        table['DETY_LO'] = a_dety_lo
        table['DETY_HI'] = a_dety_hi
        table['ENERG_LO'] = a_energy_lo
        table['ENERG_HI'] = a_energy_hi
        table['Bgd'] = a_data #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)

        table.meta['name'] = 'BACKGROUND' #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)

        return table

    def to_fits_table(self):
        """Convert cube to binary table fits format.

        Returns
        -------
        tbhdu : `~astropy.io.fits.BinTableHDU`
            Table containing the cube.
        """
        return table_to_fits_table(self.to_table())

    def to_fits_image(self):
        """Convert cube to image fits format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with:

            * one `~astropy.io.fits.PrimaryHDU` image for the cube.

            * one `~astropy.io.fits.BinTableHDU` table for the energy binning.
        """
        # data
        imhdu = fits.PrimaryHDU(data=self.data.value,
                                header=self.det_wcs.to_header())
        # add some important header information
        imhdu.header['BG_UNIT'] = '{0.unit:FITS}'.format(self.data) #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)
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
        """Write cube to fits file.

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
            Name of file to write.
        format : str, optional
            Format of the cube to write.
        kwargs
            Extra arguments for the corresponding `io.fits` `writeto` method.
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
            Array of det (X, Y) pairs to search for.

        Returns
        -------
        bin_index : `~numpy.ndarray`
            Array of integers with the indices (x, y) of the det
            bin containing the specified det (X, Y) pair.
        """
        # check that the specified det is within the boundaries of the cube
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
            Array of energies to search for.

        Returns
        -------
        bin_index : `~numpy.ndarray`
            Indices of the energy bins containing the specified energies.
        """
        # check that the specified energy is within the boundaries of the cube
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
            Array of det (X, Y) pairs to search for.

        Returns
        -------
        bin_edges : `~astropy.coordinates.Angle`
            Det bin edges (x_lo, x_hi, y_lo, y_hi).
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
            Array of energies to search for.

        Returns
        -------
        bin_edges : `~astropy.units.Quantity`
            Energy bin edges [E_min, E_max).
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
            Energy of cube bin to plot.
        ax : `~matplotlib.axes.Axes`, optional
            Axes of the figure for the plot.
        style_kwargs : dict, optional
            Style options for the plot.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes of the figure containing the plot.
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
        data = self.data[energy_bin]
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
        fig.colorbar(image, cax=cax, label='Bg rate / {}'.format(data.unit)) #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)

        # eventually close figure to avoid white canvases
        if not do_not_close_fig:
            plt.close(fig)
        return ax

    def plot_spectrum(self, det, ax=None, style_kwargs=None):
        """Plot spectra for the det bin containing the specified det (X, Y) pair.

        Parameters
        ----------
        det : `~astropy.units.Quantity`
            Det (X,Y) pair of cube bin to plot.
        ax : `~matplotlib.axes.Axes`, optional
            Axes of the figure for the plot.
        style_kwargs : dict, optional
            Style options for the plot.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes of the figure containing the plot.
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
        data = self.data[:, det_bin[1], det_bin[0]]
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
        ax.set_ylabel('Bg rate / {}'.format(data.unit)) #TODO: it could be anything (events, livetime, bg, (eventually flux for spectral cubes??!!!) pass the name as option!!! (or look it up somewhere!!!)

        # eventually close figure to avoid white canvases
        if not do_not_close_fig:
            plt.close(fig)
        return ax

    def divide_bin_volume(self):
        """Divide cube by the bin volume."""
        delta_energy = self.energy_bins[1:] - self.energy_bins[:-1]
        delta_y = self.dety_bins[1:] - self.dety_bins[:-1]
        delta_x = self.detx_bins[1:] - self.detx_bins[:-1]
        # define grid of deltas (i.e. bin widths for each 3D bin)
        delta_energy, delta_y, delta_x = np.meshgrid(delta_energy, delta_y,
                                                     delta_x, indexing='ij')
        bin_volume = delta_energy.to('MeV')*(delta_y*delta_x).to('sr') # TODO: use TeV!!!
        self.data /= bin_volume

    def set_zero_level(self):
        """Setting level 0 of the cube to something very small.

        Also for NaN values: they may appear in the 1st few E bins,
        where no stat is present: (0 events/ 0 livetime = NaN)
        """
        zero_level = Quantity(1.e-10, self.data.unit)
        zero_level_mask = self.data < zero_level
        self.data[zero_level_mask] = zero_level
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = zero_level


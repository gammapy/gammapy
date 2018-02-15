# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FOVCube container."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from ..utils.scripts import make_path
from ..utils.wcs import linear_wcs_to_arrays, linear_arrays_to_wcs
from ..utils.fits import table_to_fits_table
from ..utils.energy import Energy, EnergyBounds

__all__ = [
    'FOVCube',
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


def _parse_data_units(data_unit):
    """Utility function to parse the data units correctly."""
    # try 1st to parse them as astropy units
    try:
        u.Unit(data_unit)
    # if it fails, try to parse them as fits units
    except ValueError:
        try:
            u.Unit(data_unit, format='FITS')
        # if it fails, try to parse them as ogip units (old FITS standard)
        except ValueError:
            try:
                u.Unit(data_unit, format='ogip')
            # if it still fails, raise an exception
            except ValueError:
                raise ValueError("Invalid unit format {}.".format(data_unit))

    return data_unit


class FOVCube(object):
    """Field of view cube.

    Container class for cubes *(X, Y, energy)*.

    The class has methods for reading a cube from a FITS file,
    write a cube to a FITS file and plot the cubes among others.

    The order of the axes in the cube is **(E, y, x)**,
    so in order to access the data correctly, the call is
    ``cube.data[energy_bin, coordy_bin, coordx_bin]``.

    This class is very generic and can be used to contain cubes
    of different kinds of data. However, for the FITS reading/writing
    methods, special parameter names have to be defined, following
    the corresponding specifications.

    This is taken care of by the
    `~gammapy.background.FOVCube.define_scheme` method.
    The user only has to specify the correct **scheme** parameter.

    Currently accepted schemes are:

    * ``bg_cube``: scheme for background cubes; spatial coordinates
      *(X, Y)* are in detector coordinates (a.k.a. nominal system
      coordinates).
    * ``bg_counts_cube``: scheme for count cubes specific for
      background cube determination
    * ``bg_livetime_cube``: scheme for livetime cubes specific for
      background cube determination

    If no scheme is specified, a generic one is applied.
    New ones can be defined in
    `~gammapy.background.FOVCube.define_scheme`.
    The method also defines useful parameter names for the plots
    axis/title labels specific to each scheme.

    Parameters
    ----------
    coordx_edges : `~astropy.coordinates.Angle`, optional
        Spatial bin edges vector (low and high). X coordinate.
    coordy_edges : `~astropy.coordinates.Angle`, optional
        Spatial bin edges vector (low and high). Y coordinate.
    energy_edges : `~gammapy.utils.energy.EnergyBounds`, optional
        Energy bin edges vector (low and high).
    data : `~astropy.units.Quantity`, optional
        Data cube matrix in (energy, X, Y) format.
    scheme : str, optional
        String identifying parameter naming scheme for FITS files and plots.

    Examples
    --------
    Access cube data:

    .. code:: python

        energy_bin = cube.energy_edges.find_energy_bin('2 TeV')
        coord_bin = cube.find_coord_bin(coord=Angle([0., 0.], 'deg'))
        cube.data[energy_bin, coord_bin[1], coord_bin[0]]
    """

    scheme = ''

    def __init__(self, coordx_edges=None, coordy_edges=None, energy_edges=None, data=None, scheme=None):
        self.coordx_edges = coordx_edges
        self.coordy_edges = coordy_edges
        self._energy_edges = EnergyBounds(energy_edges)

        if data is None:
            self.data = np.zeros((len(energy_edges) - 1,
                                  len(coordy_edges) - 1,
                                  len(coordx_edges) - 1))
        else:
            self.data = data
            # TODO: make this consistent with have the 2d BCK class works
            # self.data = 'TODO'

        self.scheme = scheme

    @property
    def scheme_dict(self):
        """Naming scheme, depending on the kind of cube (dict)."""
        return self.define_scheme(self.scheme)

    @staticmethod
    def define_scheme(scheme=None):
        """Define naming scheme, depending on the kind of cube.

        Parameters
        ----------
        scheme : str, optional
            String identifying parameter naming scheme for FITS files and plots.

        Returns
        -------
        scheme_dict : dict
            Dictionary containing parameter naming scheme for FITS files and plots.
        """
        scheme_dict = dict()

        if scheme is None or scheme == '':
            # default values
            scheme_dict['hdu_fits_name'] = 'DATA'
            scheme_dict['coordx_fits_name'] = 'X'
            scheme_dict['coordy_fits_name'] = 'Y'
            scheme_dict['energy_fits_name'] = 'E'
            scheme_dict['data_fits_name'] = 'DATA'
            scheme_dict['coordx_plot_name'] = 'X'
            scheme_dict['coordy_plot_name'] = 'Y'
            scheme_dict['energy_plot_name'] = 'E'
            scheme_dict['data_plot_name'] = 'DATA'
        elif scheme == 'bg_cube':
            scheme_dict['hdu_fits_name'] = 'BACKGROUND'
            scheme_dict['coordx_fits_name'] = 'DETX'
            scheme_dict['coordy_fits_name'] = 'DETY'
            scheme_dict['energy_fits_name'] = 'ENERG'
            scheme_dict['data_fits_name'] = 'BKG'
            scheme_dict['coordx_plot_name'] = 'DET X'
            scheme_dict['coordy_plot_name'] = 'DET Y'
            scheme_dict['energy_plot_name'] = 'E'
            scheme_dict['data_plot_name'] = 'Bg rate'
        elif scheme == 'bg_counts_cube':
            scheme_dict['hdu_fits_name'] = 'COUNTS'
            scheme_dict['coordx_fits_name'] = 'DETX'
            scheme_dict['coordy_fits_name'] = 'DETY'
            scheme_dict['energy_fits_name'] = 'ENERG'
            scheme_dict['data_fits_name'] = 'COUNTS'
            scheme_dict['coordx_plot_name'] = 'DET X'
            scheme_dict['coordy_plot_name'] = 'DET Y'
            scheme_dict['energy_plot_name'] = 'E'
            scheme_dict['data_plot_name'] = 'Counts'
        elif scheme == 'bg_livetime_cube':
            scheme_dict['hdu_fits_name'] = 'LIVETIME'
            scheme_dict['coordx_fits_name'] = 'DETX'
            scheme_dict['coordy_fits_name'] = 'DETY'
            scheme_dict['energy_fits_name'] = 'ENERG'
            scheme_dict['data_fits_name'] = 'LIVETIME'
            scheme_dict['coordx_plot_name'] = 'DET X'
            scheme_dict['coordy_plot_name'] = 'DET Y'
            scheme_dict['energy_plot_name'] = 'E'
            scheme_dict['data_plot_name'] = 'Livetime'
        else:
            raise ValueError("Invalid scheme {}.".format(scheme))

        return scheme_dict

    @classmethod
    def from_fits_table(cls, hdu, scheme=None):
        """Read cube from a FITS binary table.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            HDU binary table for the cube.
        scheme : str, optional
            String identifying parameter naming scheme for FITS files and plots.

        Returns
        -------
        cube : `~gammapy.background.FOVCube`
            FOVCube object.
        """
        header = hdu.header
        data = hdu.data

        scheme_dict = cls.define_scheme(scheme)
        x_name_lo = scheme_dict['coordx_fits_name'] + '_LO'
        x_name_hi = scheme_dict['coordx_fits_name'] + '_HI'
        y_name_lo = scheme_dict['coordy_fits_name'] + '_LO'
        y_name_hi = scheme_dict['coordy_fits_name'] + '_HI'
        e_name_lo = scheme_dict['energy_fits_name'] + '_LO'
        e_name_hi = scheme_dict['energy_fits_name'] + '_HI'

        # check correct axis order: 1st X, 2nd Y, 3rd energy, 4th data
        if (header['TTYPE1'] != x_name_lo) or (header['TTYPE2'] != x_name_hi):
            raise ValueError("Expecting X axis in first 2 places, not ({0}, {1})"
                             .format(header['TTYPE1'], header['TTYPE2']))
        if (header['TTYPE3'] != y_name_lo) or (header['TTYPE4'] != y_name_hi):
            raise ValueError("Expecting Y axis in second 2 places, not ({0}, {1})"
                             .format(header['TTYPE3'], header['TTYPE4']))
        if (header['TTYPE5'] != e_name_lo) or (header['TTYPE6'] != e_name_hi):
            raise ValueError("Expecting E axis in third 2 places, not ({0}, {1})"
                             .format(header['TTYPE5'], header['TTYPE6']))
        if (header['TTYPE7'] != scheme_dict['data_fits_name']):
            raise ValueError("Expecting data axis ({0}) in fourth place, not ({1})"
                             .format(scheme_dict['data_fits_name'], header['TTYPE7']))

        # get coord X, Y binning
        coordx_edges = _make_bin_edges_array(data[x_name_lo], data[x_name_hi])
        coordy_edges = _make_bin_edges_array(data[y_name_lo], data[y_name_hi])
        if header['TUNIT1'] == header['TUNIT2']:
            coordx_unit = header['TUNIT1']
        else:
            raise ValueError("Coordinate X units not matching ({0}, {1})"
                             .format(header['TUNIT1'], header['TUNIT2']))
        if header['TUNIT3'] == header['TUNIT4']:
            coordy_unit = header['TUNIT3']
        else:
            raise ValueError("Coordinate Y units not matching ({0}, {1})"
                             .format(header['TUNIT3'], header['TUNIT4']))
        if not coordx_unit == coordy_unit:
            ss_error = "This is odd: units of X and Y coordinates not matching"
            ss_error += "({0}, {1})".format(coordx_unit, coordy_unit)
            raise ValueError(ss_error)
        coordx_edges = Angle(coordx_edges, coordx_unit)
        coordy_edges = Angle(coordy_edges, coordy_unit)

        # get energy binning
        energy_edges = _make_bin_edges_array(data[e_name_lo], data[e_name_hi])
        if header['TUNIT5'] == header['TUNIT6']:
            energy_unit = header['TUNIT5']
        else:
            raise ValueError("Energy units not matching ({0}, {1})"
                             .format(header['TUNIT5'], header['TUNIT6']))
        energy_edges = Quantity(energy_edges, energy_unit)

        # get data
        data = data[scheme_dict['data_fits_name']][0]
        data_unit = _parse_data_units(header['TUNIT7'])
        data = Quantity(data, data_unit)

        return cls(coordx_edges=coordx_edges,
                   coordy_edges=coordy_edges,
                   energy_edges=energy_edges,
                   data=data, scheme=scheme)

    @classmethod
    def from_fits_image(cls, image_hdu, energy_hdu, scheme=None):
        """Read cube from a FITS image.

        Parameters
        ----------
        image_hdu : `~astropy.io.fits.PrimaryHDU`
            FOVCube image HDU.
        energy_hdu : `~astropy.io.fits.BinTableHDU`
            Energy binning table.
        scheme : str, optional
            String identifying parameter naming scheme for FITS files and plots.

        Returns
        -------
        cube : `~gammapy.background.FOVCube`
            FOVCube object.
        """
        image_header = image_hdu.header
        energy_header = energy_hdu.header

        scheme_dict = cls.define_scheme(scheme)

        # check correct axis order: 1st X, 2nd Y, 3rd energy, 4th data
        if (image_header['CTYPE1'] != scheme_dict['coordx_fits_name']):
            raise ValueError("Expecting X axis in first place, not ({})"
                             .format(image_header['CTYPE1']))
        if (image_header['CTYPE2'] != scheme_dict['coordy_fits_name']):
            raise ValueError("Expecting Y axis in second place, not ({})"
                             .format(image_header['CTYPE2']))
        if (image_header['CTYPE3'] != scheme_dict['energy_fits_name']):
            raise ValueError("Expecting E axis in third place, not ({})"
                             .format(image_header['CTYPE3']))

        # check units
        if (image_header['CUNIT1'] != image_header['CUNIT2']):
            ss_error = "This is odd: units of X and Y coordinates not matching"
            ss_error += "({0}, {1})".format(image_header['CUNIT1'], image_header['CUNIT2'])
            raise ValueError(ss_error)
        if (image_header['CUNIT3'] != energy_header['TUNIT1']):
            ss_error = "This is odd: energy units not matching"
            ss_error += "({0}, {1})".format(image_header['CUNIT3'], energy_header['TUNIT1'])
            raise ValueError(ss_error)

        # get coord X, Y binning
        wcs = WCS(image_header, naxis=2)  # select only the (X, Y) axes
        coordx_edges, coordy_edges = linear_wcs_to_arrays(wcs,
                                                          image_header['NAXIS1'],
                                                          image_header['NAXIS2'])

        # get energy binning
        energy_edges = Quantity(energy_hdu.data['ENERGY'],
                                energy_header['TUNIT1'])

        # get data
        data = image_hdu.data
        data_unit = _parse_data_units(image_header['DATAUNIT'])
        data = Quantity(data, data_unit)

        return cls(coordx_edges=coordx_edges,
                   coordy_edges=coordy_edges,
                   energy_edges=energy_edges,
                   data=data, scheme=scheme)

    @classmethod
    def read(cls, filename, format='table', scheme=None, hdu='bkg_3d'):
        """Read cube from FITS file.

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
        scheme : str, optional
            String identifying parameter naming scheme for FITS files and plots.

        Returns
        -------
        cube : `~gammapy.background.FOVCube`
            FOVCube object.
        """
        filename = make_path(filename)
        scheme_dict = cls.define_scheme(scheme)
        hdu_list = fits.open(str(filename))

        if format == 'table':
            hdu = hdu_list[hdu]
            return cls.from_fits_table(hdu, scheme)
        elif format == 'image':
            return cls.from_fits_image(hdu_list['PRIMARY'], hdu_list['EBOUNDS'], scheme)
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
        a_coordx_lo = Quantity([self.coordx_edges[:-1]])
        a_coordx_hi = Quantity([self.coordx_edges[1:]])
        a_coordy_lo = Quantity([self.coordy_edges[:-1]])
        a_coordy_hi = Quantity([self.coordy_edges[1:]])
        a_energy_lo = Quantity([self.energy_edges[:-1]])
        a_energy_hi = Quantity([self.energy_edges[1:]])
        a_data = Quantity([self.data])

        # table
        table = Table()
        table[self.scheme_dict['coordx_fits_name'] + '_LO'] = a_coordx_lo
        table[self.scheme_dict['coordx_fits_name'] + '_HI'] = a_coordx_hi
        table[self.scheme_dict['coordy_fits_name'] + '_LO'] = a_coordy_lo
        table[self.scheme_dict['coordy_fits_name'] + '_HI'] = a_coordy_hi
        table[self.scheme_dict['energy_fits_name'] + '_LO'] = a_energy_lo
        table[self.scheme_dict['energy_fits_name'] + '_HI'] = a_energy_hi
        table[self.scheme_dict['data_fits_name']] = a_data

        table.meta['name'] = self.scheme_dict['hdu_fits_name']

        return table

    def to_fits_table(self):
        """Convert cube to binary table FITS format.

        Returns
        -------
        tbhdu : `~astropy.io.fits.BinTableHDU`
            Table containing the cube.
        """
        return table_to_fits_table(self.to_table())

    def to_fits_image(self):
        """Convert cube to image FITS format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with:

            * one `~astropy.io.fits.PrimaryHDU` image for the cube.

            * one `~astropy.io.fits.BinTableHDU` table for the energy binning.
        """
        # data
        imhdu = fits.PrimaryHDU(data=self.data.value,
                                header=self.coord_wcs.to_header())
        # add some important header information
        imhdu.header['DATAUNIT'] = '{0.unit:FITS}'.format(self.data)
        imhdu.header['CTYPE3'] = self.scheme_dict['energy_fits_name']
        imhdu.header['CUNIT3'] = '{0.unit:FITS}'.format(self.energy_edges)

        # get WCS object and write it out as a FITS header
        wcs_header = self.coord_wcs.to_header()

        # get energy values as a table HDU, via an astropy table
        energy_table = Table()
        energy_table['ENERGY'] = self.energy_edges
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
            Extra arguments for the corresponding `astropy.io.fits` ``writeto`` method.
        """
        if format == 'table':
            self.to_fits_table().writeto(outfile, **kwargs)
        elif format == 'image':
            self.to_fits_image().writeto(outfile, **kwargs)
        else:
            raise ValueError("Invalid format {}.".format(format))

    @property
    def image_extent(self):
        """Image extent (`~astropy.coordinates.Angle`).

        The output array format is ``(x_lo, x_hi, y_lo, y_hi)``.
        """
        bx = self.coordx_edges
        by = self.coordy_edges
        return Angle([bx[0], bx[-1], by[0], by[-1]])

    @property
    def spectrum_extent(self):
        """Spectrum extent (`~astropy.units.Quantity`).

        The output array format is  ``(e_lo, e_hi)``.
        """
        b = self.energy_edges
        return Quantity([b[0], b[-1]])

    @property
    def image_bin_centers(self):
        """Image bin centers **(x, y)** (2x `~astropy.coordinates.Angle`).

        Returning two separate elements for the X and Y bin centers.
        """
        coordx_bin_centers = 0.5 * (self.coordx_edges[:-1] + self.coordx_edges[1:])
        coordy_bin_centers = 0.5 * (self.coordy_edges[:-1] + self.coordy_edges[1:])
        return coordx_bin_centers, coordy_bin_centers

    @property
    def energy_edges(self):
        """Energy binning (`~gammapy.utils.energy.EnergyBounds`)."""
        return self._energy_edges

    @property
    def coord_wcs(self):
        """WCS object describing the coordinates of the coord (X, Y) bins (`~astropy.wcs.WCS`).

        This method gives the correct answer only for linear X, Y binning.
        """
        wcs = linear_arrays_to_wcs(name_x=self.scheme_dict['coordx_fits_name'],
                                   name_y=self.scheme_dict['coordy_fits_name'],
                                   bin_edges_x=self.coordx_edges,
                                   bin_edges_y=self.coordx_edges)
        return wcs

    def find_coord_bin(self, coord):
        """Find the bins that contain the specified coord (X, Y) pairs.

        Parameters
        ----------
        coord : `~astropy.coordinates.Angle`
            Array of coord (X, Y) pairs to search for.

        Returns
        -------
        bin_index : `~numpy.ndarray`
            Array of integers with the indices (x, y) of the coord
            bin containing the specified coord (X, Y) pair.
        """
        # check that the specified coord is within the boundaries of the cube
        coord_extent = self.image_extent
        check_x_lo = (coord_extent[0] <= coord[0]).all()
        check_x_hi = (coord[0] < coord_extent[1]).all()
        check_y_lo = (coord_extent[2] <= coord[1]).all()
        check_y_hi = (coord[1] < coord_extent[3]).all()
        if not (check_x_lo and check_x_hi) or not (check_y_lo and check_y_hi):
            raise ValueError("Specified coord {0} is outside the boundaries {1}."
                             .format(coord, coord_extent))

        bin_index_x = np.searchsorted(self.coordx_edges[1:], coord[0])
        bin_index_y = np.searchsorted(self.coordy_edges[1:], coord[1])

        return np.array([bin_index_x, bin_index_y])

    def find_coord_bin_edges(self, coord):
        """Find the bin edges of the specified coord (X, Y) pairs.

        Parameters
        ----------
        coord : `~astropy.coordinates.Angle`
            Array of coord (X, Y) pairs to search for.

        Returns
        -------
        bin_edges : `~astropy.coordinates.Angle`
            Coord bin edges (x_lo, x_hi, y_lo, y_hi).
        """
        bin_index = self.find_coord_bin(coord)
        bin_edges = Angle([self.coordx_edges[bin_index[0]],
                           self.coordx_edges[bin_index[0] + 1],
                           self.coordy_edges[bin_index[1]],
                           self.coordy_edges[bin_index[1] + 1]])

        return bin_edges

    def plot_image(self, energy, ax=None, style_kwargs=None):
        """Plot image for the energy bin containing the specified energy.

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`
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

        energy = Energy(energy)

        # check shape of energy: only 1 value is accepted
        if energy.size != 1:
            raise IndexError("Expected exactly 1 value for energy"
                             ", got {}.".format(energy.size))

        extent = self.image_extent

        # find energy bin containing the specified energy
        energy_bin = self.energy_edges.find_energy_bin(energy)
        energy_bin_edges = self.energy_edges[[energy_bin, energy_bin + 1]]

        # get data for the plot
        data = self.data[energy_bin]

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

        if 'cmap' not in style_kwargs:
            style_kwargs['cmap'] = 'afmhot'

        image = ax.imshow(data.value,
                          extent=extent.value,
                          origin='lower',  # do not invert image
                          interpolation='nearest',
                          **style_kwargs)

        # set title and axis names
        ax.set_title('Energy = [{0:.1f}, {1:.1f}) {2}'.format(
            energy_bin_edges[0].value, energy_bin_edges[1].value,
            energy_bin_edges.unit))
        ax.set_xlabel('{0} / {1}'.format(self.scheme_dict['coordx_plot_name'],
                                         extent.unit))
        ax.set_ylabel('{0} / {1}'.format(self.scheme_dict['coordy_plot_name'],
                                         extent.unit))

        # draw color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(image, cax=cax,
                     label='{0} / {1}'.format(self.scheme_dict['data_plot_name'],
                                              data.unit))

        # eventually close figure to avoid white canvases
        if not do_not_close_fig:
            plt.close(fig)
        return ax

    def make_spectrum(self, coord, ebounds=None):
        """
        Generate energy spectrum at a certain position in the FOV.

        Parameters
        ----------
        coord : `~astropy.units.Quantity`
            Coord (X,Y) pair of cube bin to plot.
        ebounds : `~gammapy.utils.energy.EnergyBounds`, optional
            Energy binning for the spectrum

        Returns
        -------
        spectrum : `~astropy.units.Quantity`
            Energy spectrum
        """
        ebounds = self.energy_edges if ebounds is None else ebounds
        ebins = self.energy_edges.find_energy_bin(ebounds.log_centers)

        coord = coord.flatten()
        # check shape of coord: only 1 pair is accepted
        nvalues = len(coord.flatten())
        if nvalues != 2:
            ss_error = "Expected exactly 2 values for coord (X, Y),"
            ss_error += "got {}.".format(nvalues)
            raise IndexError(ss_error)

        # find coord bin containing the specified coord coordinates
        coord_bin = self.find_coord_bin(coord)
        # get data for the plot
        spectrum = self.data[ebins, coord_bin[1], coord_bin[0]]

        return spectrum

    def plot_spectrum(self, coord, ebounds=None, ax=None, style_kwargs=None):
        """Plot spectra for the coord bin containing the specified coord (X, Y) pair.

        Parameters
        ----------
        coord : `~astropy.units.Quantity`
            Coord (X,Y) pair of cube bin to plot.
        ax : `~matplotlib.axes.Axes`, optional
            Axes of the figure for the plot.
        ebounds : `~gammapy.utils.energy.EnergyBounds`, optional
            Energy binning for the spectrum
        style_kwargs : dict, optional
            Style options for the plot.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes of the figure containing the plot.
        """
        import matplotlib.pyplot as plt

        ebounds = self.energy_edges if ebounds is None else ebounds
        data = self.make_spectrum(coord, ebounds=ebounds)
        coord_bin_edges = self.find_coord_bin_edges(coord)

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

        ax.plot(ebounds.log_centers.to('TeV'), data, drawstyle='default',
                **style_kwargs)
        ax.loglog()  # double log scale # slow!

        # set title and axis names
        ss_coordx_bin_edges = "[{0:.1f}, {1:.1f}) {2}".format(
            coord_bin_edges[0].value, coord_bin_edges[1].value,
            coord_bin_edges.unit)
        ss_coordy_bin_edges = "[{0:.1f}, {1:.1f}) {2}".format(
            coord_bin_edges[2].value, coord_bin_edges[3].value,
            coord_bin_edges.unit)

        ax.set_title('Coord = {0} {1}'.format(
            ss_coordx_bin_edges, ss_coordy_bin_edges))
        ax.set_xlabel('{0} / {1}'.format(self.scheme_dict['energy_plot_name'],
                                         ebounds.unit))
        ax.set_ylabel('{0} / {1}'.format(self.scheme_dict['data_plot_name'],
                                         data.unit))
        # eventually close figure to avoid white canvases
        if not do_not_close_fig:
            plt.close(fig)
        return ax

    @property
    def integral(self):
        """Integral of the cube (`~astropy.units.Quantity`).

        The returned quantity has dimension of the data in the cube
        times solid angle times energy.
        """
        delta_energy = self.energy_edges[1:] - self.energy_edges[:-1]
        delta_y = self.coordy_edges[1:] - self.coordy_edges[:-1]
        delta_x = self.coordx_edges[1:] - self.coordx_edges[:-1]
        # define grid of deltas (i.e. bin widths for each 3D bin)
        delta_energy, delta_y, delta_x = np.meshgrid(delta_energy, delta_y,
                                                     delta_x, indexing='ij')
        bin_volume = delta_energy * (delta_y * delta_x).to('sr')
        integral = self.data * bin_volume

        return integral.sum()

    @property
    def integral_images(self):
        """Integral of the cube images (`~astropy.units.Quantity`).

        Calculate the integral of each energy bin (slice) in the
        cube. Returns an array of integrals.

        The returned quantities have dimensions of the data in the cube
        times solid angle.
        """
        dummy_delta_energy = np.zeros_like(self.energy_edges[:-1])
        delta_y = self.coordy_edges[1:] - self.coordy_edges[:-1]
        delta_x = self.coordx_edges[1:] - self.coordx_edges[:-1]
        # define grid of deltas (i.e. bin widths for each 3D bin)
        dummy_delta_energy, delta_y, delta_x = np.meshgrid(dummy_delta_energy, delta_y,
                                                           delta_x, indexing='ij')
        bin_area = (delta_y * delta_x).to('sr')
        integral_images = self.data * bin_area

        return integral_images.sum(axis=(1, 2))

    @property
    def bin_volume(self):
        """Per-pixel bin volume.

        TODO: explain with formula and units
        """
        delta_energy = self.energy_edges[1:] - self.energy_edges[:-1]
        delta_y = self.coordy_edges[1:] - self.coordy_edges[:-1]
        delta_x = self.coordx_edges[1:] - self.coordx_edges[:-1]
        # define grid of deltas (i.e. bin widths for each 3D bin)
        delta_energy, delta_y, delta_x = np.meshgrid(delta_energy, delta_y,
                                                     delta_x, indexing='ij')
        bin_volume = delta_energy * (delta_y * delta_x).to('sr')

        return bin_volume

    def fill_events(self, event_lists):
        """Fill events histogram.

        This add the counts to the existing value array.

        Parameters
        ----------
        event_lists : list of `~gammapy.data.EventList`
           Python list of event list objects.
        """
        for event_list in event_lists:
            counts = self._fill_one_event_list(event_list)
            self.data += Quantity(counts, self.data.unit)

    def _fill_one_event_list(self, events):
        """Fill one event list into a counts array.

        Parameters
        ----------
        events :`~gammapy.data.EventList`
           Event list objects.
        """
        energy = events.energy.to('TeV').value
        detx = np.array(events.table['DETX'])
        dety = np.array(events.table['DETY'])
        sample = np.vstack([energy, detx, dety]).T

        bins = [self.energy_edges.value, self.coordy_edges.value, self.coordx_edges.value]

        hist, edges = np.histogramdd(sample, bins)

        return hist

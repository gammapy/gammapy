# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FITS utility functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .scripts import make_path
from .energy import EnergyBounds

__all__ = [
    'SmartHDUList',
    'table_to_fits_table',
    'fits_table_to_table',
    'energy_axis_to_ebounds',
]


# TODO: decide what to call this class.
# Would `FITSFile` be better than `SmartHDUList`?
class SmartHDUList(object):
    """A FITS HDU list wrapper with some sugar.

    This is a thin wrapper around `~astropy.io.fits.HDUList`,
    with some conveniences built in.

    Parameters
    ----------
    hdu_list : `~astropy.io.fits.HDUList`
        HDU list (stored in ``hdu_list`` attribute)

    Examples
    --------

    Opening a SmartHDUList calls `astropy.io.fits.open` to get a `~astropy.io.fits.HDUList`
    object, and then stores it away in the ``hdu_list`` attribute:

    >>> from gammapy.utils.fits import SmartHDUList
    >>> hdus = SmartHDUList.open('$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v08.fit.gz')
    >>> type(hdus.hdu_list)
    astropy.io.fits.hdu.hdulist.HDUList

    So of course, you can do the usual things via ``hdus.hdu_list``:

    >>> hdus.hdu_list.filename()
    >>> hdus.hdu_list.info()
    >>> [hdu.name for hdu in hdus.hdu_list]

    In addition, for a `SmartHDUList`, it's easier to get the HDUs you want:

    >>> hdus.get_hdu('Extended Sources')  # by name
    >>> hdus.get_hdu(2)  # by index
    >>> hdus.get_hdu(hdu_type='image')  # first image (skip primary if empty)
    >>> hdus.get_hdu(hdu_type='table')  # first table

    TODO: add more conveniences, e.g. to create HDU lists from lists of Gammapy
    objects that can be serialised to FITS (e.g. SkyImage, SkyCube, EventList, ...)
    """

    def __init__(self, hdu_list):
        self.hdu_list = hdu_list

    @classmethod
    def open(cls, filename, **kwargs):
        """Create from FITS file (`SmartHDUList`).

        This calls `astropy.io.fits.open`, passing ``**kwargs``.
        It reads the FITS headers, but not the data.

        The ``filename`` is passed through `~gammapy.utils.scripts.make_path`,
        which accepts strings or Path objects and does environment variable expansion.

        Parameters
        ----------
        filename : `str`
            Filename
        """
        filename = str(make_path(filename))
        hdu_list = fits.open(filename, **kwargs)
        return cls(hdu_list)

    def write(self, filename, **kwargs):
        """Write HDU list to FITS file.

        This calls `astropy.io.fits.HDUList.writeto`, passing ``**kwargs``.

        The ``filename`` is passed through `~gammapy.utils.scripts.make_path`,
        which accepts strings or Path objects and does environment variable expansion.

        Parameters
        ----------
        filename : `str`
            Filename
        """
        filename = str(make_path(filename))
        self.hdu_list.writeto(filename, **kwargs)

    @property
    def names(self):
        """List of HDU names (stripped, upper-case)."""
        return [hdu.name.strip().upper() for hdu in self.hdu_list]

    def get_hdu_index(self, hdu=None, hdu_type=None):
        """Get index of HDU with given name, number or type.

        If ``hdu`` is given, tries to find an HDU of that given name or number.
        Otherwise, if ``hdu_type`` is given, looks for the first suitable HDU.

        Raises ``KeyError`` if no suitable HDU is found.

        Parameters
        ----------
        hdu : int or str
            HDU number or name, passed to `astropy.io.fits.HDUList.index_of`.
        hdu_type : {'primary', 'image' , 'table'}
            Type of HDU to load

        Returns
        -------
        idx : int
            HDU index
        """
        # For the external API, we want the argument name to be `hdu`
        # But in this method, it's confusing because later we'll have
        # actual HDU objects. So we rename here: `hdu` -> `hdu_key`
        hdu_key = hdu
        del hdu

        if (hdu_key is None) and (hdu_type is None):
            raise ValueError('Must give either `hdu` or `hdu_type`. Got `None` for both.')

        # if (hdu_key is not None) and (hdu_type is not None):
        #     raise ValueError(
        #         'Must give either `hdu` or `hdu_type`.'
        #         ' Got a value for both: hdu={} and hdu_type={}'
        #         ''.format(hdu_key, hdu_type)
        #     )

        if hdu_key is not None:
            idx = self.hdu_list.index_of(hdu_key)
            # `HDUList.index_of` for integer input doesn't raise, just return
            # the number unchanged. Here we want to raise an error in this case.
            if not (0 <= idx < len(self.hdu_list)):
                raise KeyError('HDU not found: hdu={}. Index out of range.'.format(hdu_key))
            return idx

        if hdu_type is not None:
            for hdu_idx, hdu_object in enumerate(self.hdu_list):
                if hdu_type == 'primary':
                    if isinstance(hdu_object, fits.PrimaryHDU):
                        return hdu_idx
                elif hdu_type == 'image':
                    # The `hdu.shape` check is used to skip empty `PrimaryHDU`
                    # with no data. Those aren't very useful, now, are they?
                    if hdu_object.is_image and len(hdu_object.shape) > 0:
                        return hdu_idx
                elif hdu_type == 'table':
                    if isinstance(hdu_object, fits.BinTableHDU):
                        return hdu_idx
                else:
                    raise ValueError('Invalid hdu_type={}'.format(hdu_type))

        raise KeyError('HDU not found: hdu={}, hdu_type={}'.format(hdu_key, hdu_type))

    def get_hdu(self, hdu=None, hdu_type=None):
        """Get HDU with given name, number or type.

        This method simply calls ``get_hdu_index(hdu, hdu_type)``,
        and if successful, returns the HDU for that given index.
        """
        index = self.get_hdu_index(hdu=hdu, hdu_type=hdu_type)
        hdu = self.hdu_list[index]
        return hdu


def split_filename_hduname(location):
    """Get one HDU for a given location.

    location should be either a ``file_name`` or a file
    and HDU name in the format ``file_name[hdu_name]``.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    # TODO: Test all cases and give good exceptions / error messages
    if '[' in location:
        tokens = location.split('[')
        file_name = tokens[0]
        hdu_name = tokens[1][:-1]  # split off ']' at the end
        return fits.open(file_name)[hdu_name]
    else:
        file_name = location
        return fits.open(file_name)[0]


def fits_header_to_meta_dict(header):
    """Convert `astropy.io.fits.Header` to `~collections.OrderedDict`.

    This is a lossy conversion, only key, value is stored
    (and not e.g. comments for each FITS "card").
    Also, "COMMENT" and "HISTORY" cards are completely removed.
    """
    meta = OrderedDict(header)

    # Drop problematic header content, i.e. values of type
    # `astropy.io.fits.header._HeaderCommentaryCards`
    # Handling this well and preserving it is a bit complicated, see
    # See https://github.com/astropy/astropy/blob/master/astropy/io/fits/connect.py
    # for how `astropy.table.Table.read` does it
    # and see https://github.com/gammapy/gammapy/issues/701
    meta.pop('COMMENT', None)
    meta.pop('HISTORY', None)

    return meta


def table_to_fits_table(table, name=None):
    """Convert `~astropy.table.Table` to `astropy.io.fits.BinTableHDU`.

    The name of the table can be stored in the Table meta information
    under the ``name`` keyword.

    Additional column information ``description`` and ``ucd`` can be stored
    in the column.meta attribute and will be stored in the fits header.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table

    Returns
    -------
    hdu : `~astropy.io.fits.BinTableHDU`
        Binary table HDU
    """
    # read name and drop it from the meta information, otherwise
    # it would be stored as a header keyword in the BinTableHDU
    if name is None:
        if 'EXTNAME' in table.meta:
            name = table.meta.pop('EXTNAME', None)
        elif 'name' in table.meta:
            name = table.meta.pop('name', None)

    table.convert_unicode_to_bytestring(python3_only=True)
    data = table.as_array()

    header = fits.Header()
    header.update(table.meta)

    hdu = fits.BinTableHDU(data, header, name=name)

    # Copy over column meta-data
    for idx, colname in enumerate(table.colnames):
        # fix the order of the keywords
        hdu.header['TTYPE' + str(idx + 1)] = hdu.header.pop('TTYPE' + str(idx + 1))
        hdu.header['TFORM' + str(idx + 1)] = hdu.header.pop('TFORM' + str(idx + 1))

        if table[colname].unit is not None:
            hdu.header['TUNIT' + str(idx + 1)] = table[colname].unit.to_string('fits')

        description = table[colname].meta.get('description')
        if description:
            hdu.header['TCOMM' + str(idx + 1)] = description

        ucd = table[colname].meta.get('ucd')
        if ucd:
            hdu.header['TUCD' + str(idx + 1)] = ucd

    # TODO: this method works fine but the order of keywords in the table
    # header is not logical: for instance, list of keywords with column
    # units (TUNITi) is appended after the list of column keywords
    # (TTYPEi, TFORMi), instead of in between.
    # As a matter of fact, the units aren't yet in the header, but
    # only when calling the write method and opening the output file.
    # https://github.com/gammapy/gammapy/issues/298

    return hdu


def fits_table_to_table(hdu):
    """Convert astropy table to binary table FITS format.

    This is a generic method to convert a `~astropy.io.fits.BinTableHDU`
    to `~astropy.table.Table`.
    The name of the table is stored in the Table meta information
    under the ``name`` keyword.

    Additional column information ``description`` and ``ucd`` can will be
    read from the header and stored in the column.meta attribute.

    Parameters
    ----------
    hdu : `~astropy.io.fits.BinTableHDU`
        FITS bin table containing the astropy table columns

    Returns
    -------
    table : `~astropy.table.Table`
        astropy table containing the desired columns
    """
    data = hdu.data
    header = hdu.header
    table = Table(data, meta=header)

    # Copy over column meta-data
    for idx, colname in enumerate(hdu.columns.names):
        table[colname].unit = hdu.columns[colname].unit
        description = table.meta.pop('TCOMM' + str(idx + 1), None)
        table[colname].meta['description'] = description
        ucd = table.meta.pop('TUCD' + str(idx + 1), None)
        table[colname].meta['ucd'] = ucd

    return table


def energy_axis_to_ebounds(energy):
    """Convert `~gammapy.utils.energy.EnergyBounds` to OGIP ``EBOUNDS`` extension.

    See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2
    """
    energy = EnergyBounds(energy)
    table = Table()

    table['CHANNEL'] = np.arange(energy.nbins, dtype=np.int16)
    table['E_MIN'] = energy[:-1]
    table['E_MAX'] = energy[1:]

    hdu = table_to_fits_table(table)

    header = hdu.header
    header['EXTNAME'] = 'EBOUNDS', 'Name of this binary table extension'
    header['TELESCOP'] = 'DUMMY', 'Mission/satellite name'
    header['INSTRUME'] = 'DUMMY', 'Instrument/detector'
    header['FILTER'] = 'None', 'Filter information'
    header['CHANTYPE'] = 'PHA', 'Type of channels (PHA, PI etc)'
    header['DETCHANS'] = energy.nbins, 'Total number of detector PHA channels'
    header['HDUCLASS'] = 'OGIP', 'Organisation devising file format'
    header['HDUCLAS1'] = 'RESPONSE', 'File relates to response of instrument'
    header['HDUCLAS2'] = 'EBOUNDS', 'This is an EBOUNDS extension'
    header['HDUVERS'] = '1.2.0', 'Version of file format'

    return hdu


def ebounds_to_energy_axis(ebounds):
    """Convert ``EBOUNDS`` extension to `~gammapy.utils.energy.EnergyBounds`
    """
    table = fits_table_to_table(ebounds)
    emin = table['E_MIN'].quantity
    emax = table['E_MAX'].quantity
    energy = np.append(emin.value, emax.value[-1]) * emin.unit
    return EnergyBounds(energy)

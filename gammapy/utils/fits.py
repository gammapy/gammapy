# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
.. _utils-fits:

Gammapy FITS utilities
======================

.. _utils-fits-tables:

FITS tables
-----------

In Gammapy we use the nice `astropy.table.Table` class a lot to represent all
kinds of data (e.g. event lists, spectral points, light curves, source catalogs).
The most common format to store tables is FITS. In this section we show examples
and mention some limitations of Table FITS I/O.

Also, note that if you have the choice, you might want to use a better format
than FITS to store tables. All of these are nice and have very good support
in Astropy: ``ECSV``, ``HDF5``, ``ASDF``.

In Astropy, there is the `~astropy.table.Table` class with a nice data model
and API. Let's make an example table object that has some metadata on the
table and columns of different types:

>>> from astropy.table import Table, Column
>>> table = Table(meta={'version': 42})
>>> table['a'] = [1, 2]
>>> table['b'] = Column([1, 2], unit='m', description='Velocity')
>>> table['c'] = ['x', 'yy']
>>> table
<Table length=2>
  a     b    c
        m
int64 int64 str2
----- ----- ----
    1     1    x
    2     2   yy
>>> table.info()
<Table length=2>
name dtype unit description
---- ----- ---- -----------
   a int64
   b int64    m    Velocity
   c  str2

Writing and reading the table to FITS is easy:

>>> table.write('table.fits')
>>> table2 = Table.read('table.fits')

and works very nicely, column units and description round-trip:

>>> table2
<Table length=2>
  a      b      c
         m
int64 float64 bytes2
----- ------- ------
    1     1.0      x
    2     2.0     yy
>>> table2.info()
<Table length=2>
name  dtype  unit description
---- ------- ---- -----------
   a   int64
   b float64    m    Velocity
   c  bytes2

This is with Astropy 3.0. In older versions of Astropy this didn't use
to work, namely column description was lost.

Looking at the FITS header and ``table2.meta``, one can see that
they are cheating a bit, storing table meta in ``COMMENT``:

>>> fits.open('table.fits')[1].header
XTENSION= 'BINTABLE'           / binary table extension
BITPIX  =                    8 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   18 / length of dimension 1
NAXIS2  =                    2 / length of dimension 2
PCOUNT  =                    0 / number of group parameters
GCOUNT  =                    1 / number of groups
TFIELDS =                    3 / number of table fields
TTYPE1  = 'a       '
TFORM1  = 'K       '
TTYPE2  = 'b       '
TFORM2  = 'K       '
TUNIT2  = 'm       '
TTYPE3  = 'c       '
TFORM3  = '2A      '
VERSION =                   42
COMMENT --BEGIN-ASTROPY-SERIALIZED-COLUMNS--
COMMENT datatype:
COMMENT - {name: a, datatype: int64}
COMMENT - {name: b, unit: m, datatype: int64, description: Velocity}
COMMENT - {name: c, datatype: string}
COMMENT meta:
COMMENT   __serialized_columns__: {}
COMMENT --END-ASTROPY-SERIALIZED-COLUMNS--
>>> table2.meta
OrderedDict([('VERSION', 42),
             ('comments',
              ['--BEGIN-ASTROPY-SERIALIZED-COLUMNS--',
               'datatype:',
               '- {name: a, datatype: int64}',
               '- {name: b, unit: m, datatype: int64, description: Velocity}',
               '- {name: c, datatype: string}',
               'meta:',
               '  __serialized_columns__: {}',
               '--END-ASTROPY-SERIALIZED-COLUMNS--'])])


TODO: we'll have to see how to handle this, i.e. if we want that
behaviour or not, and how to get consistent output accross Astropy versions.
See https://github.com/astropy/astropy/issues/7364

Let's make sure for the following examples we have a clean ``table.meta``
like we did at the start:

>>> table.meta.pop('comments', None)

If you want to avoid writing to disk, the way to directly convert between
`~astropy.table.Table` and `~astropy.io.fits.BinTableHDU` is like this:

>>> hdu = fits.BinTableHDU(table)

This calls `astropy.io.fits.table_to_hdu` in ``BinTableHDU.__init__``,
i.e. if you don't pass extra options, this is equivalent to

>>> hdu = fits.table_to_hdu(table)

However, in this case, the column metadata that is serialised is
doesn't include the column ``description``.
TODO: how to get consistent behaviour and FITS headers?

>>> hdu.header
XTENSION= 'BINTABLE'           / binary table extension
BITPIX  =                    8 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   18 / length of dimension 1
NAXIS2  =                    2 / length of dimension 2
PCOUNT  =                    0 / number of group parameters
GCOUNT  =                    1 / number of groups
TFIELDS =                    3 / number of table fields
VERSION =                   42
TTYPE1  = 'a       '
TFORM1  = 'K       '
TTYPE2  = 'b       '
TFORM2  = 'K       '
TUNIT2  = 'm       '
TTYPE3  = 'c       '
TFORM3  = '2A      '

Somewhat surprisingly, ``Table(hdu)`` doesn't work and there is no
``hdu_to_table`` function; instead you have to call ``Table.read``
if you want to convert in the other direction:

>>> table2 = Table.read(hdu)
>>> table2.info()
<Table length=2>
name dtype unit
---- ----- ----
   a int64
   b int64    m
   c  str2


Another trick worth knowing about is how to read and write multiple tables
to one FITS file. There is support in the ``Table`` API to read any HDU
from a FITS file with multiple HDUs via the ``hdu`` option to ``Table.read``;
you can pass an integer HDU index or an HDU extension name string
(see :ref:`astropy:table_io_fits`).

For writing (or if you prefer also for reading) multiple tables, you should
use the in-memory conversion to HDU objects and the `~astropy.io.fits.HDUList`
like this::

    hdu_list = fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU(table, name='spam'),
        fits.BinTableHDU(table, name='ham'),
    ])
    hdu_list.info()
    hdu_list.writeto('tables.fits')


For further information on Astropy, see the Astropy docs at
:ref:`astropy:astropy-table` and :ref:`astropy:table_io_fits`.

We will have to see if / what we need here in `gammapy.utils.fits`
as a stable and nice interface on top of what Astropy provides.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import Angle, EarthLocation
from astropy.units import Quantity
from .scripts import make_path
from .energy import EnergyBounds

__all__ = ["SmartHDUList", "energy_axis_to_ebounds", "earth_location_from_dict"]


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
    >>> hdus = SmartHDUList.open('$GAMMAPY_DATA/catalogs/fermi/gll_psch_v08.fit.gz')
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
        filename : str
            Filename
        """
        filename = str(make_path(filename))
        memmap = kwargs.pop("memmap", False)
        hdu_list = fits.open(filename, memmap=memmap, **kwargs)
        return cls(hdu_list)

    def write(self, filename, **kwargs):
        """Write HDU list to FITS file.

        This calls `astropy.io.fits.HDUList.writeto`, passing ``**kwargs``.

        The ``filename`` is passed through `~gammapy.utils.scripts.make_path`,
        which accepts strings or Path objects and does environment variable expansion.

        Parameters
        ----------
        filename : str
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
            raise ValueError(
                "Must give either `hdu` or `hdu_type`. Got `None` for both."
            )

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
                raise KeyError(
                    "HDU not found: hdu={}. Index out of range.".format(hdu_key)
                )
            return idx

        if hdu_type is not None:
            for hdu_idx, hdu_object in enumerate(self.hdu_list):
                if hdu_type == "primary":
                    if isinstance(hdu_object, fits.PrimaryHDU):
                        return hdu_idx
                elif hdu_type == "image":
                    # The `hdu.shape` check is used to skip empty `PrimaryHDU`
                    # with no data. Those aren't very useful, now, are they?
                    if hdu_object.is_image and len(hdu_object.shape) > 0:
                        return hdu_idx
                elif hdu_type == "table":
                    if isinstance(hdu_object, fits.BinTableHDU):
                        return hdu_idx
                else:
                    raise ValueError("Invalid hdu_type={}".format(hdu_type))

        raise KeyError("HDU not found: hdu={}, hdu_type={}".format(hdu_key, hdu_type))

    def get_hdu(self, hdu=None, hdu_type=None):
        """Get HDU with given name, number or type.

        This method simply calls ``get_hdu_index(hdu, hdu_type)``,
        and if successful, returns the HDU for that given index.
        """
        index = self.get_hdu_index(hdu=hdu, hdu_type=hdu_type)
        hdu = self.hdu_list[index]
        return hdu


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
    meta.pop("COMMENT", None)
    meta.pop("HISTORY", None)

    return meta


def _fits_table_to_table(hdu):
    """Convert `astropy.io.fits.BinTableHDU` to `astropy.table.Table`.

    See `table_to_fits_table` to convert in the other direction and
    :ref:`utils-fits-tables` for a description and examples.

    TODO: The name of the table is stored in the Table meta information
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
    # Re-use Astropy BinTableHDU -> Table implementation
    table = Table.read(hdu)

    # In addition, copy over extra column meta-data from the HDU
    for idx, colname in enumerate(hdu.columns.names):
        idx = str(idx + 1)
        col = table[colname]

        # Unit is already handled correctly in Astropy since a long time
        # col.unit = hdu.columns[colname].unit

        description = hdu.header.pop("TCOMM" + idx, None)
        col.meta["description"] = description

        ucd = hdu.header.pop("TUCD" + idx, None)
        col.meta["ucd"] = ucd

    return table


def energy_axis_to_ebounds(energy):
    """Convert `~gammapy.utils.energy.EnergyBounds` to OGIP ``EBOUNDS`` extension.

    See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2
    """
    energy = EnergyBounds(energy)
    table = Table()

    table["CHANNEL"] = np.arange(energy.nbins, dtype=np.int16)
    table["E_MIN"] = energy[:-1]
    table["E_MAX"] = energy[1:]

    hdu = fits.BinTableHDU(table)

    header = hdu.header
    header["EXTNAME"] = "EBOUNDS", "Name of this binary table extension"
    header["TELESCOP"] = "DUMMY", "Mission/satellite name"
    header["INSTRUME"] = "DUMMY", "Instrument/detector"
    header["FILTER"] = "None", "Filter information"
    header["CHANTYPE"] = "PHA", "Type of channels (PHA, PI etc)"
    header["DETCHANS"] = energy.nbins, "Total number of detector PHA channels"
    header["HDUCLASS"] = "OGIP", "Organisation devising file format"
    header["HDUCLAS1"] = "RESPONSE", "File relates to response of instrument"
    header["HDUCLAS2"] = "EBOUNDS", "This is an EBOUNDS extension"
    header["HDUVERS"] = "1.2.0", "Version of file format"

    return hdu


def ebounds_to_energy_axis(ebounds):
    """Convert ``EBOUNDS`` extension to `~gammapy.utils.energy.EnergyBounds`
    """
    table = Table.read(ebounds)
    emin = table["E_MIN"].quantity
    emax = table["E_MAX"].quantity
    energy = np.append(emin.value, emax.value[-1]) * emin.unit
    return EnergyBounds(energy)


# TODO: add unit test
def earth_location_from_dict(meta):
    """Create `~astropy.coordinates.EarthLocation` from FITS header dict."""
    lon = Angle(meta["GEOLON"], "deg")
    lat = Angle(meta["GEOLAT"], "deg")
    # TODO: should we support both here?
    # Check latest spec if ALTITUDE is used somewhere.
    if "GEOALT" in meta:
        height = Quantity(meta["GEOALT"], "meter")
    elif "ALTITUDE" in meta:
        height = Quantity(meta["ALTITUDE"], "meter")
    else:
        raise KeyError("The GEOALT or ALTITUDE header keyword must be set")

    return EarthLocation(lon=lon, lat=lat, height=height)

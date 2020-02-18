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

However, in this case, the column metadata that is serialized is
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
import numpy as np
from astropy.coordinates import Angle, EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity


__all__ = ["energy_axis_to_ebounds", "earth_location_from_dict", "LazyFitsData"]


class LazyFitsData(object):
    """A lazy FITS data descriptor.

    Parameters
    ----------
    cache : bool
        Whether to cache the data.
    """

    def __init__(self, cache=True):
        self.cache = cache

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, objtype):
        if instance is None:
            # Accessed on a class, not an instance
            return self

        value = instance.__dict__.get(self.name)
        if value is not None:
            return value
        else:
            hdu_loc = instance.__dict__.get(self.name + "_hdu")
            value = hdu_loc.load()
            if self.cache:
                instance.__dict__[self.name] = value
            return value

    def __set__(self, instance, value):
        from gammapy.data import HDULocation

        if isinstance(value, HDULocation):
            instance.__dict__[self.name + "_hdu"] = value
        else:
            instance.__dict__[self.name] = value


def energy_axis_to_ebounds(energy):
    """Convert `~astropy.units.Quantity` to OGIP ``EBOUNDS`` extension.

    See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2
    """
    energy = Quantity(energy)
    table = Table()

    table["CHANNEL"] = np.arange(len(energy) - 1, dtype=np.int16)
    table["E_MIN"] = energy[:-1]
    table["E_MAX"] = energy[1:]

    hdu = fits.BinTableHDU(table)

    header = hdu.header
    header["EXTNAME"] = "EBOUNDS", "Name of this binary table extension"
    header["TELESCOP"] = "DUMMY", "Mission/satellite name"
    header["INSTRUME"] = "DUMMY", "Instrument/detector"
    header["FILTER"] = "None", "Filter information"
    header["CHANTYPE"] = "PHA", "Type of channels (PHA, PI etc)"
    header["DETCHANS"] = len(energy) - 1, "Total number of detector PHA channels"
    header["HDUCLASS"] = "OGIP", "Organisation devising file format"
    header["HDUCLAS1"] = "RESPONSE", "File relates to response of instrument"
    header["HDUCLAS2"] = "EBOUNDS", "This is an EBOUNDS extension"
    header["HDUVERS"] = "1.2.0", "Version of file format"

    return hdu


def ebounds_to_energy_axis(ebounds):
    """Convert ``EBOUNDS`` extension to `~astropy.units.Quantity`
    """
    table = Table.read(ebounds)
    emin = table["E_MIN"].quantity
    emax = table["E_MAX"].quantity
    energy = np.append(emin.value, emax.value[-1]) * emin.unit
    return energy


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

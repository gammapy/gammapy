# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table
from ..utils.scripts import make_path
from ..utils.nddata import BinnedDataAxis, NDDataArray

__all__ = [
    'InstrumentResponse',
]


class InstrumentResponse(object):
    """Class with an NDDataArray containing an instrument response.

    Attributes
    ----------
    names :  list of str
        Names of the axes of the NDDataArray
    data : NDDataArray
        The data array containing the insturment response

    Methods
    -------
    evaluate(point={'axis_name' : value})
        Evaulate the data and fixed point


    Examples
    -------
    Read and evaluate the effective area from a FITS file.
    >>> from gammapy.irf import InstrumentResponse
    >>> irf = InstrumentResponse.from_fits(path, extension='EFFECTIVE AREA')
    >>> point = {'THETA': 3.5*u.deg, 'ENERG': 1*u.TeV}
    >>> interpolated_result = irf.evaluate(point)
    """
    names = None
    interpolation_modes = None

    def __init__(self, data):
        self._data = data

    def __repr__(self):
        return repr(self._data)

    @classmethod
    def make_class(cls, config):
        class MyClass(cls):
            pass

        return MyClass

    @property
    def axis_names(self):
        return self._data.axes.names

    def evaluate(self, **kwargs):
        """Evaluate the instrument response at the given point.

        Parameters
        -----------
        point: a dict containing the coordinates where you want to evaluate the response

        Returns
        -----------
        A number with an associated unit.

        Examples
        --------
        Evaluate instrument response at a point

        >>> point = {'DETX': 3.5*u.deg, 'DETY': 3.5*u.deg, 'ENERG': 1*u.TeV}
        >>> irf.evaluate(point)
        """
        return self._data.evaluate(**kwargs)

    @classmethod
    def read(cls, filename, hdu):
        """Read from file."""
        filename = make_path(filename)
        table = Table.read(str(filename), hdu=hdu)
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table):
        """
        Read the instrument response from an astropy table

        Parameters
        -----------
        table : astropy.table.Table
            the table containing the instrument response

        Examples
        --------
        >>> table = Table.read(path)
        >>> irf = InstrumentResponse.from_table(table)
        """
        # read table data that stores n-dimensional data in ogip convention
        bounds = table.colnames[:-1]
        low_bounds = bounds[::2]
        high_bounds = bounds[1::2]

        data = table[table.colnames[-1]].quantity[0].T

        # we need to check this unit specifically because ctools writes weird units
        # into fits tables and astropy goes haywire
        if data.unit == '1/s/MeV/sr':
            import astropy.units as u
            data = data.value * u.Unit('1/(s MeV sr)')

        if not cls.names:
            names = [n.replace('_LO', '') for n in low_bounds]

        if not cls.interpolation_modes:
            interpolation_modes = ['linear'] * len(names)

        axes = []
        for colname_low, colname_high, name, mode in zip(low_bounds, high_bounds, cls.names, cls.interpolation_modes):
            low = np.ravel(table[colname_low]).quantity
            high = np.ravel(table[colname_high]).quantity
            axis = BinnedDataAxis(low, high, interpolation_mode=mode, name=name)
            axes.append(axis)

        return cls(NDDataArray(axes=axes, data=data))


class Background3D(InstrumentResponse):
    """Background 3D IRF.

    Data format specification: :ref:`gadf:bkg_3d`

    This is a prototype, let's see if it can replace
    `gammapy.irf.background.Background3D`
    """
    names = ['detx', 'dety', 'energy']
    interpolation_modes = ['linear', 'linear', 'log']

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function, unicode_literals
import itertools
import numpy as np
import abc
from ..extern.bunch import Bunch
from astropy.units import Quantity
from astropy.table import Table, Column
from astropy.extern import six
from .array import array_stats_str

__all__ = [
    'NDDataArray',
    'DataAxis',
    'BinnedDataAxis',
]

# Note: test for this class are implemented in
# gammapy/irf/tests/test_effective_area2.py.

six.add_metaclass(abc.ABCMeta)
class NDDataArray(object):
    """ND Data Array Base class

    TODO: Document
    """
    axis_names = ()
    """Axis names. This specifies the axis order"""

    interp_kwargs = dict(bounds_error=False)
    """Interpolation kwargs used to initialize the
    `scipy.interpolate.RegularGridInterpolator`.  The interpolation behaviour
    of an individual axis ('log', 'linear') can be passed to the axis on
    initialization."""

    def __init__(self, **kwargs):

        # TODO : Dynamically generate function signature
        # https://github.com/astropy/astropy/blob/ffc0a89b2c42fd440eb19bcb2f93db90cab3c98b/astropy/utils/codegen.py#L30
        data = kwargs.pop('data', None)

        meta = kwargs.pop('meta', None)
        if meta is not None:
            self.meta = Bunch(meta)

        self._axes = list()
        for axis_name in self.axis_names:
            value = kwargs.pop(axis_name)
            axis = getattr(self, axis_name)
            axis.data = Quantity(value)
            self._axes.append(axis)

        self.data = data
        self._regular_grid_interp = None

    @property
    def axes(self):
        """Array holding the axes in correct order"""
        return self._axes

    @property
    def data(self):
        """Array holding the n-dimensional data"""
        return self._data

    @data.setter
    def data(self, data):
        """Set data

        Some sanitiy checks are performed to avoid an invalid array

        Parameters
        ----------
        data : `~astropy.units.Quantity`, array-like
            Data array
        """
        dimension = len(data.shape)
        if dimension != self.dim:
            raise ValueError('Overall dimensions to not match. '
                             'Data: {}, Hist: {}'.format(dimension, self.dim))

        for dim in np.arange(self.dim):
            axis = self.axes[dim]
            if axis.nbins != data.shape[dim]:
                msg = 'Data shape does not match in dimension {d}\n'
                msg += 'Axis {n} : {sa}, Data {sd}'
                raise ValueError(msg.format(d=dim, n=self.axis_names[dim],
                                            sa=axis.nbins, sd=data.shape[dim]))
        self._data = data

    @property
    def dim(self):
        """Dimension (number of axes)"""
        return len(self.axes)

    def to_table(self):
        """Convert to `~astropy.table.Table`"""

        raise NotImplementedError("Broken")

        pairs = [_table_columns_from_data_axis(a) for a in self.axes[::-1]]
        cols = [_ for pair in pairs for _ in pair]
        cols.append(Column(data=[self.data.value], name='data', unit=self.data.unit))
        table = Table(cols)
        return table

    def write(self, *args, **kwargs):
        """Write to disk

        Calling astropy I/O interface
        see http://docs.astropy.org/en/stable/io/unified.html
        """
        self.to_table().write(*args, **kwargs)

    @classmethod
    def from_table(cls, table):
        """Fits Reader"""
        raise NotImplementedError('')

    @classmethod
    def read(cls, *args, **kwargs):
        """Read from disk

        Calling astropy I/O interface
        see http://docs.astropy.org/en/stable/io/unified.html
        """
        table = Table.read(*args, **kwargs)
        return cls.from_table(table)

    def __str__(self):
        """String representation"""
        ss = 'Data array summary info\n'
        for axis, axname in zip(self.axes, self.axis_names):
            ss += array_stats_str(axis.data, axname)
        ss += array_stats_str(self.data, 'Data')
        return ss

    def evaluate(self, method='linear', **kwargs):
        """Evaluate NDData Array

        This function provides a uniform interface to several interpolators.
        The evaluation nodes are given as ``kwargs``.

        Currently available:
        `~scipy.interpolate.RegularGridInterpolator`, methods: linear, nearest

        Parameters
        ----------
        method : str {'linear', 'nearest'}
            Interpolation method
        kwargs : dict
            Keys are the axis names, Values the evaluation points

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values, axis order is the same as for the NDData array
        """

        values = list()
        for axname, axis in zip(self.axis_names, self.axes):
            # Extract values for each axis, default: nodes
            temp = kwargs.pop(axname, axis.nodes)
            # Transform to correct unit
            temp = temp.to(axis.unit).value
            # Transform to match interpolation behaviour of axis
            values.append(np.atleast_1d(axis._interp_values(temp)))

        if method == 'linear':
            return self._eval_regular_grid_interp(
                values, method='linear') * self.data.unit
        elif method == 'nearest':
            return self._eval_regular_grid_interp(
                values, method='nearest') * self.data.unit
        else:
            raise ValueError('Interpolator {} not available'.format(method))

    def _eval_regular_grid_interp(self, values, method='linear'):
        """Evaluate linear interpolator

        Input: list of values to evaluate, in correct units and correct order.
        """
        if self._regular_grid_interp is None:
            self._add_regular_grid_interp()

        shapes = np.append(*[np.shape(_) for _ in values])
        # Flatten in order to support 2D array input
        values = [_.flatten() for _ in values]
        points = list(itertools.product(*values))
        res = self._regular_grid_interp(points, method=method)
        res = np.reshape(res, shapes).squeeze()

        return res

    def _add_regular_grid_interp(self):
        """Add `~scipy.interpolate.RegularGridInterpolator`


        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.RegularGridInterpolator.html
        """
        from scipy.interpolate import RegularGridInterpolator

        points = [a._interp_nodes() for a in self.axes]
        values = self.data.value

        self._regular_grid_interp = RegularGridInterpolator(points, values,
                                                            **self.interp_kwargs)


class DataAxis(object):
    """Data axis to be used with NDDataArray

    Axis values are interpreted as nodes.
    """
    def __init__(self, data=None, interpolation_mode='linear'):
        self.data = data
        self._interpolation_mode = interpolation_mode

    @property
    def unit(self):
        """Axis unit"""
        return self.data.unit

    @classmethod
    def logspace(cls, vmin, vmax, nbins, unit=None):
        """Create axis with equally log-spaced nodes

        if no unit is given, it will be taken from vmax

        Parameters
        ----------
        vmin : `~astropy.units.Quantity`, float
            Lowest value
        emax : `~astropy.units.Quantity`, float
            Highest value
        bins : int
            Number of bins
        unit : `~astropy.units.UnitBase`, str
            Unit
        """

        if unit is not None:
            vmin = Quantity(vmin, unit)
            vmax = Quantity(vmax, unit)
        else:
            vmin = Quantity(vmin)
            vmax = Quantity(vmax)
            unit = vmax.unit
            vmin = vmin.to(unit)

        x_min, x_max = np.log10([vmin.value, vmax.value])
        vals = np.logspace(x_min, x_max, nbins)

        return cls(vals * unit, interpolation_mode='log')

    def find_node(self, val):
        """Find next node

        Parameters
        ----------
        val : `~astropy.units.Quantity`
            Lookup value
        """
        val = Quantity(val)

        if not val.unit.is_equivalent(self.unit):
            raise ValueError('Units {} and {} do not match'.format(
                val.unit, self.unit))

        val = val.to(self.data.unit)
        val = np.atleast_1d(val)
        x1 = np.array([val] * self.nbins).transpose()
        x2 = np.array([self.nodes] * len(val))
        temp = np.abs(x1 - x2)
        idx = np.argmin(temp, axis=1)
        return idx

    @property
    def nbins(self):
        """Number of bins"""
        return self.data.size

    @property
    def nodes(self):
        """Evaluation nodes"""
        return self.data

    @property
    def interpolation_mode(self):
        """Interpolation mode
        """
        return self._interpolation_mode

    def _interp_nodes(self):
        """Nodes to be used for interpolation"""
        if self.interpolation_mode == 'log':
            return np.log10(self.nodes.value)
        else:
            return self.nodes.value

    def _interp_values(self, values):
        """Transform values correctly for interpolation"""
        if self.interpolation_mode == 'log':
            return np.log10(values)
        else:
            return values


class BinnedDataAxis(DataAxis):
    """Data axis for binned axis

    Axis values are interpreted as bin edges
    """
    @classmethod
    def logspace(cls, emin, emax, nbins, unit=None):
        return super(BinnedDataAxis, cls).logspace(
            emin, emax, nbins + 1, unit)

    @property
    def nbins(self):
        """Number of bins"""
        return self.data.size - 1

    @property
    def nodes(self):
        """Evaluation nodes

        Depending on the interpolation mode, either log or lin center are
        returned
        """
        if self.interpolation_mode == 'log':
            return self.log_center()
        else:
            return self.lin_center()

    def lin_center(self):
        """Linear bin centers"""
        return (self.data[:-1] + self.data[1:]) / 2

    def log_center(self):
        """Logarithmic bin centers"""
        return np.sqrt(self.data[:-1] * self.data[1:])


def _table_columns_from_data_axis(axis):
    """Helper function to translate a data axis to two table columns

    The first column contains the lower bounds, the second the upper bounds.
    This satisfies the format definition here
    http://gamma-astro-data-formats.readthedocs.io/en/latest/info/fits-arrays.html
    """
    # BROKEN!
    if isinstance(axis, BinnedDataAxis):
        data_hi = axis.data.value[1:]
        data_lo = axis.data.value[:-1]
    elif isinstance(axis, DataAxis):
        data_hi = axis.data.value
        data_lo = axis.data.value
    else:
        raise ValueError('Invalid axis type')

    c_hi = Column(data=[data_hi], unit=axis.unit, name='{}_HI'.format(axis.name))
    c_lo = Column(data=[data_lo], unit=axis.unit, name='{}_LO'.format(axis.name))

    return c_lo, c_hi

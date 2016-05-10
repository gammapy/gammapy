"""Scratch for NDDataArray class"""

import itertools
import numpy as np
from astropy.units import Quantity, Unit
from astropy.table import Table, Column


class NDDataArray(object):
    """ND Data Array

    This class represents an ND Data Array. The data stored as numpy array
    attribute. The data axis are separate classes and this class has them as
    members. The axis order follows numpy convention for arrays, i.e. the axis
    added last is at index 0. For an example see NDData_demo.ipynb.
    """
    def __init__(self):
        self._axes = list()
        self._data = None
        # Todo: Should this be set on construction?
        self._lininterp = None

    def add_axis(self, axis):
        """Add axis

        This data array is set to None to avoid unwanted behaviour.

        Parameters
        ----------
        axis : `DataAxis`
            axis
        """
        default_names = {0: 'x', 1: 'y', 2: 'z'}
        if axis.name is None:
            axis.name = default_names[self.dim]
        self._axes = [axis] + self._axes

        # Quick solution: delete data to prevent unwanted behaviour
        self._data = None
        self._lininterp = None

    @property
    def axes(self):
        """Array holding all axes"""
        return self._axes

    @property
    def data(self):
        """Array holding the ND data"""
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
        data = np.array(data)
        d = len(data.shape)
        if d != self.dim:
            raise ValueError('Overall dimensions to not match. '
                             'Data: {}, Hist: {}'.format(d, self.dim))

        for dim in np.arange(self.dim):
            if self.axes[dim].nbins != data.shape[dim]:
                a = self.axes[dim]
                raise ValueError('Data shape does not match in dimension {d}\n'
                                 'Axis "{n}": {sa}, Data {sd}'.format(
                                 d=dim, n=a.name, sa=a.nbins,
                                 sd=data.shape[dim]))

        self._data = Quantity(data)

    @property
    def axis_names(self):
        """Currently set axis names"""
        return [a.name for a in self.axes]
    
    def get_axis_index(self, name):
        """Return axis index by it name

        Parameters
        ----------
        name : str
            Valid axis name
        """
        for a in self.axes:
            if a.name == name:
                return self.axes.index(a)
        raise ValueError("No axis with name {}".format(name))

    def get_axis(self, name):
        """Return axis by it name

        Parameters
        ----------
        name : str
            Valid axis name
        """
        idx = self.get_axis_index(name)
        return self.axes[idx]

    @property
    def dim(self):
        """Dimension (number of axes)"""
        return len(self.axes)

    def to_table(self):
        """Convert to astropy.Table"""

        pairs = [_table_columns_from_data_axis(a) for a in self.axes]
        cols = [_ for pair in pairs for _ in pair]
        cols.append(Column(data=[self.data.value], name='data', unit=self.data.unit))
        t = Table(cols)
        return t

    def write(self, *args, **kwargs):
        """Write to disk

        Calling astropy I/O interface
        see http://docs.astropy.org/en/stable/io/unified.html
        """
        self.to_table().write(*args, **kwargs)

    @classmethod
    def from_table(cls, table):
        """Create from astropy table

        The table must represent the convention at
        http://gamma-astro-data-formats.readthedocs.io/en/latest/info/fits-arrays.html#bintable-hdu

        Parameters
        ----------
        table : `~astropy.table`
            table
        """
        nddata = cls()
        cols = table.columns
        data = cols.pop(cols.keys()[-1])
        col_pairs = zip(cols[::2].values(), cols[1::2].values())
        axes = [_data_axis_from_table_columns(cl, ch) for cl, ch in col_pairs]
        nddata._axes = axes
        nddata.data = data.squeeze()
        return nddata

    @classmethod
    def read(cls, *args, **kwargs):
        """Read from disk

        Calling astropy I/O interface
        see http://docs.astropy.org/en/stable/io/unified.html
        """
        t = Table.read(*args, **kwargs)
        return cls.from_table(t)

    def __str__(self):
        """String representation"""
        return str(self.to_table())

    def find_node(self, **kwargs):
        """Find nearest node

        Parameters
        ----------
        kwargs : dict
            Search values
        """
        for key in kwargs.keys():
            if key not in self.axis_names:
                raise ValueError('No axis for key {}'.format(key))

        for name, val in zip(self.axis_names, self.axes):
            kwargs.setdefault(name, val.nodes)

        nodes = list()
        for a in self.axes:
            value = kwargs[a.name]
            nodes.append(a.find_node(value))

        return nodes

    def evaluate_nearest(self, **kwargs):
        """Evaluate NDData Array

        No interpolation, this is equivalent to ``evaluate(method='nearest')``
        and will probably go away at some point.

        Parameters
        ----------
        kwargs : dict
            Axis names are keys, Quantity array are values
        """
        idx = self.find_node(**kwargs)
        data = self.data
        for i in np.arange(self.dim):
            data = np.take(data, idx[i], axis=i)

        return data

    def evaluate(self, method='linear', **kwargs):
        """Evaluate NDData Array

         This function provides a uniform interface to several interpolators

        Interpolators have to be added before this function can be used.
        TODO : Do we want a default interpolator to be added on construction?

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
        for key in kwargs.keys():
            if key not in self.axis_names:
                raise ValueError('No axis for key {}'.format(key))

        # Use nodes on unspecified axes
        for name, val in zip(self.axis_names, self.axes):
            kwargs.setdefault(name, val.nodes)

        # Put in correct units
        for key in kwargs:
            val = Quantity(kwargs[key], unit=self.get_axis(key).unit)
            kwargs[key] = np.atleast_1d(val).value

        # Bring in correct order
        values = [kwargs[_] for _ in self.axis_names]

        # Apply log10 where necessary
        # Todo : only calc log10 when needed
        log_values = [np.log10(_) for _ in values]
        loginterp = [a.log_interpolation for a in self.axes]
        values = np.where(loginterp, log_values, values)

        if method == 'linear':
            return self._eval_linear(values, method='linear') * self.data.unit
        elif method == 'nearest':
            return self._eval_linear(values, method='nearest') * self.data.unit
        else:
            raise ValueError('Interpolator {} not available'.format(method))

    def add_linear_interpolator(self, **kwargs):
        """Add `~scipy.interpolate.RegularGridInterpolator`

        The interpolation behaviour of an individual axis can be adjusted by
        setting the ``log_interpolation`` attribute to True. After that the
        interpolator has to be setup anew. TODO: Make this somehow more
        convenient?

        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.RegularGridInterpolator.html
        """
        kwargs.setdefault('bounds_error', False)

        from scipy.interpolate import RegularGridInterpolator

        points = [a._interp_nodes() for a in self.axes]
        values = self.data.value

        self._lininterp = RegularGridInterpolator(points, values, **kwargs)

    def _eval_linear(self, values, method='linear'):
        """Evaluate linear interpolator

        Input: list of values to evaluate, in correct units and correct order.
        """
        if self._lininterp is None:
            raise ValueError('Linear interpolation requested but no linear '
                             'interpolator initialized')

        shapes = [np.shape(_)[0] for _ in values]
        points = list(itertools.product(*values))
        res = self._lininterp(points, method=method)
        res = np.reshape(res, shapes)

        return res

    def plot_image(self, ax=None, plot_kwargs = {}, **kwargs):
        """Plot image

        Only avalable for 2D (after slicing)
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        data = self.evaluate(**kwargs)
        if len(data.squeeze().shape) != 2:
            raise ValueError('Data has shape {} after slicing. '
                             'Need 2d data for image plot'.format(data.shape))

        ax.set_xlabel('{} [{}]'.format(self.axes[0].name, self.axes[0].unit))
        ax.set_ylabel('{} [{}]'.format(self.axes[1].name, self.axes[1].unit))
        ax.imshow(data.transpose(), origin='lower', **plot_kwargs)

    def plot_profile(self, axis, ax=None, **kwargs):
        """Show data as function of one axis

        Parameters
        ----------
        axis : DataAxis
            data axis to use
        """

        raise NotImplementedError

        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        ax_ind = self.get_axis_index(axis)
        kwargs.setdefault(axis, self.axes[ax_ind])

        x = kwargs.pop(axis)

        y = self.evaluate(**kwargs)


class DataAxis(Quantity):
    """Data axis for unbinned values"""
    def __new__(cls, energy, unit=None, dtype=None, copy=True, name=None):
        self = super(DataAxis, cls).__new__(cls, energy, unit,
                                            dtype=dtype, copy=copy)

        self.name = name
        self.log_interpolation = False
        return self

    def __array_finalize__(self, obj):
        super(DataAxis, self).__array_finalize__(obj)

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

        val = val.to(self.unit)
        val = np.atleast_1d(val)
        x1 = np.array([val] * self.nbins).transpose()
        x2 = np.array([self.nodes] * len(val))
        temp = np.abs(x1 - x2)
        idx = np.argmin(temp, axis=1)
        return idx

    @property
    def nbins(self):
        """Number of bins"""
        return self.size

    @property
    def nodes(self):
        """Evaluation nodes"""
        return self

    def _interp_nodes(self):
        """Nodes to be used for interpolation"""
        if not self.log_interpolation:
            return self.nodes.value
        else:
            return np.log10(self.nodes.value)


class BinnedDataAxis(DataAxis):
    """Data axis for binned values"""
    @classmethod
    def linspace(cls, min, max, nbins, unit=None):
        """Create linearly spaced axis"""
        if unit is None:
            raise NotImplementedError

        data = np.linspace(min, max, nbins+1)
        unit = Unit(unit)
        return cls(data, unit)

    @property
    def nbins(self):
        """Number of bins"""
        return self.size - 1

    @property
    def nodes(self):
        """Evaluation nodes"""
        return self.lin_center()

    def lin_center(self):
        """Linear bin centers"""
        return DataAxis(self[:-1] + self[1:]) / 2


def _data_axis_from_table_columns(col_lo, col_hi):
    """Helper function to translate two table columns to a data axis"""
    if (col_lo.data == col_hi.data).all():
        return DataAxis(col_lo.data[0], unit=col_lo.unit, name=col_lo.name[:-3])
    else:
        data = np.append(col_lo.data[0], col_hi.data[0][-1])
        return BinnedDataAxis(data, unit=col_lo.unit, name=col_lo.name[:-3])


def _table_columns_from_data_axis(axis):
    """Helper function to translate a data axis to two table columns

    The first column contains the lower bounds, the second the upper bounds.
    This satisfies the format definition here
    http://gamma-astro-data-formats.readthedocs.io/en/latest/info/fits-arrays.html
    """

    if isinstance(axis, BinnedDataAxis):
        data_hi = axis.value[1:]
        data_lo = axis.value[:-1]
    elif isinstance(axis, DataAxis):
        data_hi = axis.value
        data_lo = axis.value
    else:
        raise ValueError('Invalid axis type')

    c_hi = Column(data=[data_hi], unit=axis.unit, name='{}_HI'.format(axis.name))
    c_lo = Column(data=[data_lo], unit=axis.unit, name='{}_LO'.format(axis.name))

    return c_lo, c_hi


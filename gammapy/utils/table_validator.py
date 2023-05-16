# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import namedtuple
from collections.abc import MutableMapping
from astropy.table import Column, Table
from astropy.units import UnitTypeError
import yaml

GADF_EVENT_TABLE_DEFINITION = """
EVENT_ID: { dtype: int, required: true, unit: null}
TIME:     { dtype: float, required: true, unit: s}
RA:       { dtype: float, required: true, unit: deg}
DEC:      { dtype: float, required: true, unit: deg}
ENERGY:   { dtype: float, required: true, unit: TeV}
EVENT_TYPE: { dtype: int8 }
MULTIP :  { dtype: int }
GLON :    { dtype: float, unit: deg}
GLAT :    { dtype: float, unit: deg}
ALT :     { dtype: float, unit: deg}
AZ :      { dtype: float, unit: deg}
DETX :    { dtype: float, unit: deg}
DETY :    { dtype: float, unit: deg}
THETA :   { dtype: float, unit: deg}
PHI :     { dtype: float, unit: deg}
GAMMANESS : { dtype: float}
DIR_ERR : { dtype: float, unit: deg}
ENERGY_ERR : { dtype: float, unit: TeV}
COREX :    { dtype: float, unit: m}
COREY :    { dtype: float, unit: m}
CORE_ERR : { dtype: float, unit: m}
XMAX :     { dtype: float, unit: m}
XMAX_ERR : { dtype: float, unit: m}
HIL_MSW :  { dtype: float, unit: ''}
HIL_MSW_ERR : { dtype: float, unit: ''}
HIL_MSL : { dtype: float, unit: ''}
HIL_MSL_ERR : { dtype: float, unit: ''}
"""

ColumnDefinition = namedtuple(
    "ColumnDefinition",
    ["dtype", "unit", "shape", "description", "required"],
    defaults=[None, None, (), str, False],
)


class ColumnValidator(ColumnDefinition):
    def validate_column(self, column):
        """Check that input Column satisfies the column definition."""
        self.check_column_type(self.dtype, column)
        self.check_column_shape(self.shape, column)
        self.check_column_unit(self.unit, column)
        return column

    @staticmethod
    def check_column_type(dtype, column):
        if column.dtype.name == dtype:
            return True
        else:
            raise TypeError(
                f"Column dtype incorrect. Expected {dtype}, got {column.dtype} instead."
            )

    @staticmethod
    def check_column_shape(shape, column):
        if column.shape[1:] == shape:
            return True
        else:
            raise TypeError(
                f"Column shape incorrect. Expected {shape}, got {column.shape} instead."
            )

    @staticmethod
    def check_column_unit(unit, column):
        unit = unit if unit is not None else ""
        if column.unit is None and unit == "":
            return True
        elif column.unit.is_equivalent(unit):
            return True
        else:
            raise UnitTypeError(
                f"Column unit incorrect. Expected {unit}, got {column.unit} instead."
            )

    def to_column(self, name):
        """Build column from column definition.

        Parameters
        ----------
        name : str
            the `~astropy.table.Column` name

        Returns
        -------
        column : `~astropy.table.Column`
        """
        return Column(
            name=name,
            dtype=self.dtype,
            unit=self.unit,
            shape=self.shape,
            description=self.description,
        )


class TableValidator(MutableMapping):
    def __init__(self, **kwargs):
        self._data = {}
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        del self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, ColumnDefinition):
            self._data[key] = value
        else:
            raise TypeError(f"Invalid type: {type(value)!r}")

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def to_yaml(self):
        res = {}
        for key in self._data:
            res[key] = self._data[key]._asdict()
        return yaml.dump(
            res, sort_keys=False, indent=4, width=80, default_flow_style=False
        )

    @classmethod
    def from_yaml(cls, yaml_str):
        coldefs = yaml.safe_load(yaml_str)
        res = cls()
        for key, item in coldefs.items():
            res[key] = ColumnValidator(**item)
        return res

    @property
    def required_columns(self):
        req = []
        for key in self._data:
            if self[key].required:
                req.append(key)
        return req

    @property
    def optional_columns(self):
        req = []
        for key in self._data:
            if not self[key].required:
                req.append(key)
        return req

    def run(self, table):
        for key in self.required_columns:
            self[key].validate_column(table[key])
        for key in self.optional_columns:
            if key in table.colnames:
                self[key].validate_column(table[key])
        return table

    def to_table(self, include_optional=None):
        """Build empty table from columns definition.

        Parameters
        ----------
        include_optional : str or list
            List of optional columns to include.
            If 'all', include all columns. Default is None.

        Returns
        -------
        table : `~astropy.table.Table`
            the empty table
        """
        data = {}
        if include_optional is None:
            include_optional = []
        elif include_optional == "all":
            include_optional = self.optional_columns

        for key in self._data:
            if key in include_optional or self[key].required:
                data[key] = self[key].to_column(name=key)
        return Table(data)

    @classmethod
    def from_builtin(cls, format=GADF_EVENT_TABLE_DEFINITION):
        """Create validator from pre-defined format definition"""
        return cls.from_yaml(format)

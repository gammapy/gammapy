# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Format validation of serialized data products.

``DefinitionValidator`` checks one header (keywords) or one table (columns)
against one definition ``{name: spec}``, as described in
`gammapy.io.fits.registries`. It has no knowledge of data formats or versions.

Table columns are checked through ``ColumnDescription`` objects, built from an
in-memory table (``columns_from_table``) or from a FITS binary-table header
(``columns_from_header``), in which case the table data is not read.
"""

import re
from typing import NamedTuple, Optional

import numpy as np
import astropy.units as u

__all__ = [
    "ColumnDescription",
    "DefinitionValidator",
    "columns_from_header",
    "columns_from_table",
]

# TFORMn: repeat count, type code and, for variable-length arrays, the element
# type code, e.g. "6E", "1PE(100)"
_TFORM = re.compile(r"^\s*(\d*)\s*([A-Za-z])([A-Za-z]?)")


# --------------- DEFINITION VALIDATOR ---------------
def _value(meta, key):
    """Keyword value, with empty values ("" / [] / None) treated as absent."""
    val = meta.get(key)
    if val is None or (isinstance(val, (str, list)) and len(val) == 0):
        return None
    return val


_FITS_DTYPES = {
    "str": (str,),
    "int": (int, np.integer),
    "float": (float, int, np.floating, np.integer),
    "bool": (bool, np.bool_),
}


def _has_dtype(value, dtype):
    dtypes = (dtype,) if isinstance(dtype, str) else tuple(dtype)
    for name in dtypes:
        if name in ("int", "float") and isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, _FITS_DTYPES[name]):
            return True
    return False


def _check_keyword(name, value, spec):
    """Errors of one header keyword value against its spec (dtype, allowed)."""
    errors = []
    dtype = spec.get("dtype")
    if dtype and not _has_dtype(value, dtype):
        errors.append(f"Invalid {name}={value!r}: expected {dtype}")
    allowed = spec.get("allowed")
    if allowed is not None and value not in allowed:
        errors.append(f"Invalid {name}={value!r}: allowed values {list(allowed)}")
    return errors


class ColumnDescription(NamedTuple):
    """What the column checks need to know about a column.

    Attributes
    ----------
    ndim : int
        Number of dimensions per row.
    unit : `~astropy.units.UnitBase` or None
        Column unit.
    dtype : `numpy.dtype` or None
        Data type of the values (of the array elements for a variable-length
        array column); None if unknown.
    bit_field : bool or None
        True for a FITS bit-field column (TFORM ``X``), False for any other FITS
        column, None if unknown: an in-memory table does not keep the TFORM, and
        astropy reads both ``X`` and ``L`` columns as booleans.
    """

    ndim: int
    unit: Optional[u.UnitBase]
    dtype: Optional[np.dtype] = None
    bit_field: Optional[bool] = None


def _table_column_dtype(column):
    """Value dtype of a table column; element dtype for variable-length arrays."""
    if column.dtype.kind == "O":
        for value in column:
            if value is not None and hasattr(value, "dtype"):
                return value.dtype
        return None
    return column.dtype


def columns_from_table(table):
    """Column descriptions of an in-memory `~astropy.table.Table`."""
    return {
        name: ColumnDescription(
            ndim=table[name].ndim - 1,
            unit=table[name].unit,
            dtype=_table_column_dtype(table[name]),
        )
        for name in table.colnames
    }


# TFORM type codes -> numpy dtype, as read by astropy.io.fits (FITS standard,
# Pence et al. 2010, Table 18). "X" (bit) and "A" (string) are handled apart.
_TFORM_DTYPES = {
    "L": "bool",
    "B": "uint8",
    "I": "int16",
    "J": "int32",
    "K": "int64",
    "E": "float32",
    "D": "float64",
    "C": "complex64",
    "M": "complex128",
}

# TZERO offsets for non-default integer types (FITS standard, Pence et al. 2010,
# Table 19): signed bytes and unsigned 16/32/64-bit integers. Any other TZERO /
# TSCAL gives scaled float64 values. Note: astropy reads a "B" column with
# TZERO=-128 as float64; the standard (and this check) treat it as int8.
_INTEGER_TZERO = {
    "B": (-128, "int8"),
    "I": (2**15, "uint16"),
    "J": (2**31, "uint32"),
    "K": (2**63, "uint64"),
}


def _header_column_dtype(code, inner, tzero, tscal):
    """Value dtype of a binary table column from its TFORM / TZERO / TSCAL."""
    if code in ("P", "Q"):  # variable-length array: dtype of its elements
        code = inner
    if code == "A":
        return np.dtype("S")
    if code == "X":
        return np.dtype("bool")
    if code not in _TFORM_DTYPES:
        return None
    scaled = (tzero not in (None, 0)) or (tscal not in (None, 1))
    if scaled and code in "BIJK":
        offset, integer = _INTEGER_TZERO[code]
        if tzero == offset and tscal in (None, 1):
            return np.dtype(integer)
        return np.dtype("float64")
    return np.dtype(_TFORM_DTYPES[code])


def _tdim(value):
    """Parse a TDIMn value such as ``"(3,2)"`` into a tuple of ints."""
    return tuple(int(x) for x in str(value).strip().strip("()").split(",") if x.strip())


def columns_from_header(header):
    """Column descriptions of a binary table, read from its FITS header only.

    Uses the ``TFIELDS``, ``TTYPEn``, ``TFORMn``, ``TDIMn``, ``TUNITn``,
    ``TZEROn`` and ``TSCALn`` keywords, so no table data is read. The per-row
    ``ndim``, the unit and the dtype are those `~astropy.table.Table.read` gives
    the column; ``bit_field`` tells bit-field columns (TFORM ``X``) apart.
    """
    columns = {}
    for idx in range(1, int(header.get("TFIELDS", 0)) + 1):
        name = header.get(f"TTYPE{idx}")
        if name is None:
            continue
        match = _TFORM.match(str(header.get(f"TFORM{idx}", "")))
        repeat = int(match.group(1) or 1) if match else 1
        code = match.group(2).upper() if match else ""
        inner = match.group(3).upper() if match else ""
        tdim = header.get(f"TDIM{idx}")
        if code in ("P", "Q"):  # variable-length array: one object per row
            ndim = 0
        elif code == "A":  # strings: TDIM includes the string length
            ndim = len(_tdim(tdim)) - 1 if tdim else 0
        elif code == "X":  # bits are read as a boolean array
            ndim = 1
        elif tdim:
            ndim = len(_tdim(tdim))
        else:
            ndim = 1 if repeat > 1 else 0
        tunit = header.get(f"TUNIT{idx}")
        unit = (
            u.Unit(tunit, format="fits", parse_strict="silent")
            if tunit not in (None, "")
            else None
        )
        dtype = _header_column_dtype(
            code, inner, header.get(f"TZERO{idx}"), header.get(f"TSCAL{idx}")
        )
        columns[str(name)] = ColumnDescription(
            ndim=ndim, unit=unit, dtype=dtype, bit_field=code == "X"
        )
    return columns


_COLUMN_KINDS = {
    "int": lambda dt: np.issubdtype(dt, np.integer),
    "float": lambda dt: np.issubdtype(dt, np.floating),
    "bool": lambda dt: dt.kind == "b",
    "str": lambda dt: dt.kind in "SU",
}


def _column_has_dtype(column, name):
    """Whether a column matches one dtype name of a column spec.

    * "int", "float", "bool", "str": any type of that kind (any width);
    * "bit": a FITS bit field (TFORM ``X``); for an in-memory table, where the
      TFORM is not known, any boolean column;
    * any other name, e.g. "int64", "float64": that exact numpy type.
    """
    dtype = column.dtype
    if name == "bit":
        if column.bit_field is not None:
            return column.bit_field
        return dtype is not None and dtype.kind == "b"
    if column.bit_field:  # a bit field matches only "bit"
        return False
    if name in _COLUMN_KINDS:
        return _COLUMN_KINDS[name](dtype)
    return dtype.newbyteorder("=") == np.dtype(name)


def _dtype_name(column):
    if column.bit_field:
        return "bit"
    if column.dtype.kind in "SU":
        return "str"
    return column.dtype.name


def _check_column(name, column, spec):
    """Errors of one column (a `ColumnDescription`) against its spec.

    Checks ``ndim``, ``dtype`` (a name or a list of accepted names, see
    `_column_has_dtype`) and ``unit``. A spec without ``unit`` expects a
    dimensionless column; units are compared by equivalence.
    """
    errors = []
    ndim = spec.get("ndim")
    if ndim is not None and column.ndim != ndim:
        errors.append(
            f"{name}: Column ndim incorrect. Expected {ndim}, got {column.ndim}."
        )
    dtype = spec.get("dtype")
    if dtype is not None and column.dtype is not None:
        names = [dtype] if isinstance(dtype, str) else list(dtype)
        try:
            ok = any(_column_has_dtype(column, n) for n in names)
        except TypeError as e:
            return errors + [f"{name}: invalid dtype {dtype!r} in definition: {e}"]
        if not ok:
            expected = names[0] if len(names) == 1 else names
            errors.append(
                f"{name}: Column dtype incorrect. "
                f"Expected {expected}, got {_dtype_name(column)}."
            )
    try:
        unit = spec.get("unit")
        expected = u.one if unit is None else u.Unit(unit)
    except ValueError as e:
        return errors + [
            f"{name}: invalid unit {spec.get('unit')!r} in definition: {e}"
        ]
    found = u.one if column.unit is None else column.unit
    if not found.is_equivalent(expected):
        errors.append(
            f"{name}: Column unit incorrect. Expected {expected}, got {column.unit}."
        )
    return errors


class DefinitionValidator:
    """Check a header or a table against a definition ``{name: spec}``.

    The same definition layout serves header keywords and table columns (see
    `gammapy.io.fits.registries`). Both share the presence rules (``required``,
    ``required_if``, ``required_unless``, the conditions being read from the
    header); the per-item checks differ:

    * keywords: ``dtype`` (Python type of the value) and ``allowed``;
    * columns: ``ndim``, ``dtype`` (numpy type of the values) and ``unit``.
    """

    def __init__(self, definition):
        self.definition = definition

    @classmethod
    def from_dict(cls, definition):
        return cls(definition)

    def check_header(self, header):
        """Errors of a header (dict) against the definition; empty if compliant."""
        errors = []
        missing = [
            name
            for name, spec in self.definition.items()
            if spec.get("required") and _value(header, name) is None
        ]
        if missing:
            errors.append(f"Missing mandatory keyword(s): {missing}")
        errors += self._check_conditions(
            lambda name: _value(header, name) is not None, header
        )
        for name, spec in self.definition.items():
            value = _value(header, name)
            if value is not None:
                errors += _check_keyword(name, value, spec)
        return errors

    def check_table(self, table):
        """Errors of an in-memory table's columns; empty if compliant."""
        return self.check_columns(columns_from_table(table), table.meta)

    def check_columns(self, columns, header=None):
        """Errors of column descriptions ``{name: ColumnDescription}``.

        Build them with `columns_from_table` or `columns_from_header`.
        ``header`` gives the values of the ``required_if`` / ``required_unless``
        conditions.
        """
        errors = []
        header = header or {}
        required = [n for n, s in self.definition.items() if s.get("required")]
        optional = [n for n, s in self.definition.items() if not s.get("required")]
        missing = [name for name in required if name not in columns]
        if missing:
            errors.append(f"Missing mandatory column(s): {missing}")
        for name in required + optional:
            if name in columns:
                errors += _check_column(name, columns[name], self.definition[name])
        errors += self._check_conditions(lambda name: name in columns, header)
        return errors

    def _check_conditions(self, present, header):
        """``required_if`` / ``required_unless``, grouped by condition."""
        conditional = {}
        for name, spec in self.definition.items():
            for rule, negate in (("required_if", False), ("required_unless", True)):
                condition = spec.get(rule)
                if not condition:
                    continue
                holds = all(_value(header, k) == v for k, v in condition.items())
                if holds != negate and not present(name):
                    desc = ", ".join(f"{k}={_value(header, k)!r}" for k in condition)
                    conditional.setdefault(desc, []).append(name)
        return [f"{names} required when {desc}" for desc, names in conditional.items()]

    def with_defaults(self, header):
        """Copy of ``header`` with absent keywords set to their ``default`` (defaults first)."""
        out = {
            name: spec["default"]
            for name, spec in self.definition.items()
            if "default" in spec and _value(header, name) is None
        }
        out.update({k: v for k, v in dict(header).items() if k not in out})
        return out

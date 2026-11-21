# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Data-format registries: versioned column and header-keyword definitions.

Each format module (e.g. ``gammapy.io.fits.registries.gadf``) describes its format as
plain dicts, and the format is registered in ``DATA_FORMATS_MODELS``:

* ``"TABLE"``          version -> HDU class key -> {column: spec}
* ``"HEADER"``         version -> HDU class key -> {keyword: spec}
* ``"HDU_CLASS_KEY"``  fn(header) -> HDU class key                  (optional)
* ``"HEADER_CHECKS"``  [fn(header, class_key, version) -> list[str]] (optional; for
  rules that cannot be expressed in the keyword spec)

Other modules may add further entries to a format (e.g. ``"READER_WRITER"``);
validation only reads the ones above.

A header definition is {keyword: spec}. All spec entries are optional:

* ``dtype``            "str" | "int" | "float" | "bool", or a list of them
* ``required``         True if the keyword is mandatory
* ``required_if``      {other_keyword: value}: mandatory when all conditions hold
* ``required_unless``  {other_keyword: value}: mandatory unless all conditions hold
* ``allowed``          list of accepted values
* ``default``          value filled in on write when the keyword is absent
* ``unit``, ``comment``  documentation; not checked

Keywords absent from the definition are allowed (headers are open). An HDU
class key with no header (or no table) definition is reported as an error.

A table definition is {column: spec}, with the same presence entries
(``required``, ``required_if``, ``required_unless``, conditions read from the
header) and:

* ``dtype``            type of the values, or a list of accepted ones:
  "int" | "float" | "bool" | "str" (any width), "bit" (FITS bit field,
  TFORM ``X``), or an exact numpy type name such as "int64" or "float64".
  Give it only where the format specifies the column type.
* ``ndim``             number of dimensions per row
* ``unit``             expected unit; equivalent units are accepted, and a column
  without ``unit`` in its spec must be dimensionless
* ``description``      documentation; not checked

Columns absent from the definition are allowed.

This package holds data only: it imports nothing from ``gammapy.data``,
``gammapy.irf`` or ``gammapy.maps``.
"""

import copy

DATA_FORMATS_MODELS: dict = {}

DEFAULT_DATA_FORMAT = "GADF"
DEFAULT_DATA_FORMAT_VERSION = "0.3"


def _kw(dtype, **spec):
    """Keyword spec shorthand."""
    return {"dtype": dtype, **spec}


def compose_header(*blocks, required=(), updates=None):
    """Build a header definition from keyword blocks.

    Parameters
    ----------
    *blocks : dict
        Keyword blocks, merged in order (later blocks win).
    required : iterable of str
        Keywords to mark as mandatory.
    updates : dict, optional
        {keyword: spec updates}, applied last (e.g. defaults, conditions).
    """
    out = {}
    for block in blocks:
        out.update(copy.deepcopy(block))
    for keyword in required:
        out.setdefault(keyword, {})["required"] = True
    for keyword, spec in (updates or {}).items():
        out.setdefault(keyword, {}).update(spec)
    return out


# --------------- BUILT-IN FORMATS ---------------
# Each module registers itself in DATA_FORMATS_MODELS on import. Add new
# formats here (they are pure data, so importing them is cheap).
from gammapy.io.fits.registries import gadf  # noqa: E402, F401

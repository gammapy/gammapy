# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This packages contains python packages that are bundled with the affiliated
package but are external to the affiliated package, and hence are developed in
a separate source tree. Note that this package is distinct from the /cextern
directory of the source code distribution, as that directory only contains C
extension code.

The following Python packages were bundled with Gammapy.
* ``xmltodict.py`` for easily converting XML from / to Python dicts
  Origin: https://github.com/martinblech/xmltodict/blob/master/xmltodict.py
* ``validator.py`` for `~astropy.units.Quantity` validation helper functions.
  Origin: https://github.com/astrofrog/sedfitter/blob/master/sedfitter/utils/validator.py
* ``pathlib.py`` is a copy pathlib2 from github on 2015-10-20
  Origin https://raw.githubusercontent.com/mcmtroffaes/pathlib2/develop/pathlib2.py
  One line was patched: `import six` -> `from astropy.extern import six`
  See also: https://pypi.python.org/pypi/pathlib2/
* ``appdirs.py`` is a copy from Github on 2016-01-08
  Origin: https://raw.githubusercontent.com/ActiveState/appdirs/master/appdirs.py
  See also: https://pypi.python.org/pypi/appdirs/
"""

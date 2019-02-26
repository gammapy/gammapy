# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains `extern` code, i.e. code that we just copied
here into the `gammapy.extern` package, because we wanted to use it,
but not have an extra dependency (these are single-file external packages).

Alphabetical order:

* ``appdirs.py`` is a copy from Github on 2016-01-08
  Origin: https://raw.githubusercontent.com/ActiveState/appdirs/master/appdirs.py
  See also: https://pypi.org/project/appdirs/
* ``validator.py`` for `~astropy.units.Quantity` validation helper functions.
  Origin: https://github.com/astrofrog/sedfitter/blob/master/sedfitter/utils/validator.py
* ``xmltodict.py`` for easily converting XML from / to Python dicts
  Origin: https://github.com/martinblech/xmltodict/blob/master/xmltodict.py
* ``zeros.py`` - Modified copy of the file from ``scipy.optimize``,
  used by the TS map computation code.
* ``skimage.py`` - utility functions copied from scikit-image
  They have the same BSD license as we do.
"""

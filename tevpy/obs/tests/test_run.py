# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .. import run


def test_Run():
    run.Run(GLON=42, GLAT=43)


def test_RunList():
    run.RunList()

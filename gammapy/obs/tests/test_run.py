# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ...obs import Run, RunList


def test_Run():
    Run(GLON=42, GLAT=43)


def test_RunList():
    RunList()

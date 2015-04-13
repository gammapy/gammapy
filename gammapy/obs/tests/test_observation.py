# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ...obs import Observation, ObservationTable


def test_Observation():
    Observation(GLON=42, GLAT=43)


def test_ObservationTable():
    ObservationTable()

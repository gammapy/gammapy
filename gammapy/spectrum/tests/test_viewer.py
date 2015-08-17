# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ..viewer import SEDViewer


def test_SEDViewer():
    sed_viewer = SEDViewer()
    sed_viewer.run()

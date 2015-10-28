# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_data
from ..hess import SourceCatalogHGPS


@requires_data('hgps')
class TestSourceCatalogHGPS:

    def setup(self):
        self.cat = SourceCatalogHGPS()

    def test_something(self):
        assert self.cat.name == 'hgps'
        assert len(self.cat.table) == 66

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...utils.testing import requires_data
from ..hess import SourceCatalogHGPS


@requires_data('hgps')
class TestSourceCatalogHGPS:
    def setup(self):
        self.cat = SourceCatalogHGPS()
        self.cat.hdu_list.info()

    def test_source_table(self):
        assert self.cat.name == 'hgps'
        assert len(self.cat.table) == 77

    def test_component_table(self):
        assert len(self.cat.components) == 97

    def test_associations_table(self):
        assert len(self.cat.associations) == 216


@requires_data('hgps')
class TestSourceCatalogObjectHGPS:
    def setup(self):
        self.cat = SourceCatalogHGPS()
        # Use HESS J1825-137 as a test source
        self.source_name = 'HESS J1825-137'
        self.source = self.cat[self.source_name]

    def test_single_gauss(self):
        source = self.cat['HESS J1930+188']
        assert source.data['Spatial_Model'] == 'Gaussian'

    def test_multi_gauss(self):
        source = self.cat['HESS J1825-137']
        assert source.data['Spatial_Model'] == '3-Gaussian'

    def test_snr(self):
        source = self.cat['HESS J1713-397']
        assert source.data['Spatial_Model'] == 'Shell'

    def test_name(self):
        assert self.source.name == self.source_name

    def test_index(self):
        assert self.source.index == 52

    def test_data(self):
        data = self.source.data
        assert data['Source_Class'] == 'PWN'

    def test_pprint(self):
        self.source.pprint()

    def test_print_info(self):
        self.source.print_info()

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy import units as u
from ...utils.testing import requires_data
from ..catalog import CatalogImageEstimator
from ...image import SkyImage
from ...catalog import (SourceCatalog3FHL, SourceCatalogGammaCat, SourceCatalogHGPS,
                        SourceCatalog3FGL)


class TestCatalogImageEstimator(object):
    @requires_data('gammapy-extra')
    @requires_data('gamma-cat')
    def test_flux_gammacat(self):
        reference = SkyImage.empty(xref=18.0, yref=-0.6, nypix=41,
                                   nxpix=41, binsz=0.1)

        filename = '$GAMMAPY_EXTRA/datasets/catalogs/gammacat/gammacat.fits.gz'
        catalog = SourceCatalogGammaCat(filename)
        estimator = CatalogImageEstimator(reference=reference,
                                          emin=1 * u.TeV,
                                          emax=1E4 * u.TeV)

        result = estimator.run(catalog)

        actual = result['flux'].data.sum()
        selection = catalog.select_image_region(reference)

        assert len(selection.table) == 3

        desired = selection.table['spec_flux_1TeV'].sum()
        assert_allclose(actual, desired, rtol=1E-3)

    @requires_data('gammapy-extra')
    def test_flux_3FHL(self):
        reference = SkyImage.empty(xref=18.0, yref=-0.6, nypix=81,
                                   nxpix=81, binsz=0.1)

        catalog = SourceCatalog3FHL()
        estimator = CatalogImageEstimator(reference=reference,
                                          emin=10 * u.GeV,
                                          emax=1000 * u.GeV)

        result = estimator.run(catalog)

        actual = result['flux'].data.sum()
        selection = catalog.select_image_region(reference)

        assert len(selection.table) == 7

        desired = selection.table['Flux'].sum()
        assert_allclose(actual, desired, rtol=1E-2)

    @requires_data('gammapy-extra')
    def test_flux_3FGL(self):
        reference = SkyImage.empty(xref=18.0, yref=-0.6, nypix=81,
                                   nxpix=81, binsz=0.1)

        catalog = SourceCatalog3FGL()
        estimator = CatalogImageEstimator(reference=reference,
                                          emin=1 * u.GeV,
                                          emax=100 * u.GeV)

        result = estimator.run(catalog)

        actual = result['flux'].data.sum()
        selection = catalog.select_image_region(reference)

        assert len(selection.table) == 18

        desired = selection.table['Flux1000'].sum()
        assert_allclose(actual, desired, rtol=1E-2)

    @requires_data('hgps')
    def test_flux_hgps(self):
        reference = SkyImage.empty(xref=18.0, yref=-0.6, nypix=81,
                                   nxpix=81, binsz=0.1)

        catalog = SourceCatalogHGPS()
        estimator = CatalogImageEstimator(reference=reference,
                                          emin=1 * u.TeV,
                                          emax=1000 * u.TeV)

        result = estimator.run(catalog)

        actual = result['flux'].data.sum()
        selection = catalog.select_image_region(reference)

        assert len(selection.table) == 7

        desired = selection.table['Flux_Spec_Int_1TeV'].sum()
        assert_allclose(actual, desired, rtol=1E-2)

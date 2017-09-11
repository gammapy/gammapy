# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.wcs import WCS
from astropy.tests.helper import remote_data
from ...utils.testing import requires_dependency, requires_data
from ..catalog import CatalogImageEstimator, catalog_image, catalog_table, _source_image
from ...image import SkyImage
from ...irf import EnergyDependentTablePSF
from ...cube import SkyCube
from ...datasets import FermiGalacticCenter
from ...spectrum import LogEnergyAxis
from ...catalog import (SourceCatalog3FHL, SourceCatalogGammaCat, SourceCatalogHGPS,
                        SourceCatalog3FGL)


def test_extended_image():
    # TODO: implement me
    pass


@remote_data
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_source_image():
    reference_hdu = SkyImage.empty(nxpix=10, nypix=10, binsz=1).to_image_hdu()
    reference_wcs = WCS(reference_hdu.header)
    energy_axis = LogEnergyAxis(u.Quantity([10, 500], 'GeV'))
    reference = SkyCube(data=reference_hdu.data,
                        wcs=reference_wcs, energy_axis=energy_axis)

    psf_file = FermiGalacticCenter.filenames()['psf']
    psf = EnergyDependentTablePSF.read(psf_file)

    image, energies = _source_image(catalog='1FHL',
                                    reference_cube=reference,
                                    total_flux=True)

    actual = image.sum()
    # Flux of sources within a 10x10 deg region about Galactic Center
    expected = 1.6098631760996795e-07
    assert_allclose(actual, expected)


@remote_data
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_catalog_image():
    reference_hdu = SkyImage.empty(nxpix=10, nypix=10, binsz=1).to_image_hdu()
    reference_wcs = WCS(reference_hdu.header)
    energy = u.Quantity([10, 500], 'GeV')

    psf_file = FermiGalacticCenter.filenames()['psf']
    psf = EnergyDependentTablePSF.read(psf_file)

    out_cube = catalog_image(reference_hdu, psf, catalog='1FHL',
                             source_type='point', total_flux=True,
                             sim_table=None)

    actual = out_cube.data.sum()

    # Ensures flux is consistent following PSF convolution to within 1%
    expected = 1.6098631760996795e-07
    assert_allclose(actual, expected, rtol=0.01)


@remote_data
@requires_data('gammapy-extra')
def test_catalog_table():
    # Checks catalogs are loaded correctly

    table_1fhl = catalog_table('1FHL')
    assert len(table_1fhl) == 514

    table_2fgl = catalog_table('2FGL')
    assert len(table_2fgl) == 1873


class TestCatalogImageEstimator(object):
    @requires_data('gammapy-extra')
    def test_flux_gammacat(self):
        reference = SkyImage.empty(xref=18.0, yref=-0.6, nypix=41,
                                   nxpix=41, binsz=0.1)

        catalog = SourceCatalogGammaCat()
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
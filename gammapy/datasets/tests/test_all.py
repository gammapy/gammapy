# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.utils.data import get_pkg_data_filename
from astropy.units import Quantity
from astropy.io import fits
from astropy.tests.helper import remote_data
from .. import poisson_stats_image, FermiGalacticCenter
from .. import fetch_fermi_catalog
from ...utils.testing import assert_quantity


def test_poisson_stats_image():
    """Get the data file via the gammapy.data.poisson_stats_image function"""
    data = poisson_stats_image()
    assert data.sum() == 40896


def test_poisson_stats_image_direct():
    """Get the data file directly via get_pkg_data_filename"""
    filename = get_pkg_data_filename('../poisson_stats_image/counts.fits.gz')
    data = fits.getdata(filename)
    assert data.sum() == 40896


def test_poisson_stats_extra_info():
    images = poisson_stats_image(extra_info=True)
    refs = dict(counts=40896, model=41000, source=1000, background=40000)
    for name, expected in refs.items():
        assert_allclose(images[name].sum(), expected)


class TestFermiGalacticCenter():
    
    def test_filenames(self):
        filenames = FermiGalacticCenter.filenames()
        assert isinstance(filenames, dict)

    def test_psf(self):
        psf = FermiGalacticCenter.psf()
        assert psf['PSF'].data.shape == (20,)
        assert psf['THETA'].data.shape == (300,)

    def test_counts(self):
        counts = FermiGalacticCenter.counts()
        assert counts.data.shape == (201, 401)
        assert counts.data.sum() == 24803

    def test_diffuse_model(self):
        diffuse_model = FermiGalacticCenter.diffuse_model()
        assert diffuse_model.data.shape == (30, 21, 61)
        assert_quantity(diffuse_model.energy[0], Quantity(50, 'MeV'))


@remote_data
def test_fetch_fermi_catalog():
    n_hdu = len(fetch_fermi_catalog('2FGL'))
    assert(n_hdu, 5)
    
    n_sources = len(fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog'))
    assert(n_sources, 1873)

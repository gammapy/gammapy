# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import remote_data
from ... import datasets


@remote_data
def test_load_catalog_atnf(tmpdir):
    catalog = datasets.load_catalog_atnf()
    assert len(catalog) == 2399

    # Check if catalog can be serialised to FITS
    filename = str(tmpdir.join('atnf_test.fits'))
    catalog.write(filename)


# TODO: activate test when available
@remote_data
def _test_load_catalog_hess_galactic(tmpdir):
    catalog = datasets.load_catalog_hess_galactic()
    assert len(catalog) == 42

    # Check if catalog can be serialised to FITS
    filename = str(tmpdir.join('hess_test.fits'))
    catalog.write(filename)


@remote_data
def test_load_catalog_green(tmpdir):
    catalog = datasets.load_catalog_green()
    assert len(catalog) == 294

    # Check if catalog can be serialised to FITS
    filename = str(tmpdir.join('green_test.fits'))
    catalog.write(filename)


@remote_data
def test_load_catalog_snrcat(tmpdir):
    snrcat = datasets.fetch_catalog_snrcat()

    # Check SNR table
    table = snrcat.snr_table
    assert len(table) > 300
    expected_colnames = ['Source_Name', 'RAJ2000']
    assert set(expected_colnames).issubset(table.colnames)
    # Check if catalog can be serialised to FITS
    filename = str(tmpdir.join('snrcat_test.fits'))
    table.write(filename)

    # Check OBS table
    table = snrcat.obs_table
    print(table.colnames)
    assert len(table) > 1000
    expected_colnames = ['SNR_id', 'source_id']
    assert set(expected_colnames).issubset(table.colnames)
    # Check if catalog can be serialised to FITS
    filename = str(tmpdir.join('obs_test.fits'))
    table.write(filename)


@remote_data
def test_load_catalog_tevcat(tmpdir):
    catalog = datasets.load_catalog_tevcat()
    assert len(catalog) == 173

    # Check if catalog can be serialised to FITS
    filename = str(tmpdir.join('tevcat_test.fits'))
    catalog.write(filename)

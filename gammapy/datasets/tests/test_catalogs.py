# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from tempfile import NamedTemporaryFile
from astropy.tests.helper import remote_data
from ... import datasets


@remote_data
def test_load_catalog_atnf():
    catalog = datasets.load_catalog_atnf()
    assert len(catalog) == 2399

    # Check if catalog can be serialised to FITS
    filename = NamedTemporaryFile(suffix='.fits').name
    catalog.write(filename)


# TODO: activate test when available
@remote_data
def _test_load_catalog_hess_galactic():
    catalog = datasets.load_catalog_hess_galactic()
    assert len(catalog) == 42

    # Check if catalog can be serialised to FITS
    filename = NamedTemporaryFile(suffix='.fits').name
    catalog.write(filename)


@remote_data
def test_load_catalog_green():
    catalog = datasets.load_catalog_green()
    assert len(catalog) == 294

    # Check if catalog can be serialised to FITS
    filename = NamedTemporaryFile(suffix='.fits').name
    catalog.write(filename)


@remote_data
def test_load_catalog_snrcat():
    catalog = datasets.fetch_catalog_snrcat()
    assert len(catalog) > 300
    assert set(['Source_Name', 'RAJ2000']).issubset(catalog.colnames)

    # Check if catalog can be serialised to FITS
    filename = NamedTemporaryFile(suffix='.fits').name
    catalog.write(filename)


@remote_data
def test_load_catalog_tevcat():
    catalog = datasets.load_catalog_tevcat()
    assert len(catalog) == 173

    # Check if catalog can be serialised to FITS
    filename = NamedTemporaryFile(suffix='.fits').name
    catalog.write(filename)

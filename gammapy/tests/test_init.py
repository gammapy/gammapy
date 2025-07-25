# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammapy integration and system tests.

This package can be used for tests that involve several
Gammapy sub-packages, or that don't fit anywhere else.
"""

import gammapy


def test_get_acknowledgment():
    text = gammapy.__acknowledgment__
    assert "Astropy" in text
    assert "iminuit" in text


def test_bibtex():
    text = gammapy.__bibtex__
    for item in ["article", "author", "DOI", "title", "journal"]:
        assert item in text
    assert "https://doi.org/10.1051/0004-6361/202346488" in text

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
# from ... import hspec

try:
    import sherpa
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False


@pytest.mark.skipif('not HAS_SHERPA')
def test_sherpa_imports():
    import sherpa
    import sherpa.astro.ui as sau
    from group import grpGetGroupSum


@pytest.mark.skipif('not HAS_SHERPA')
def test_hspec_imports():
    from ...hspec import load_model
    from ...hspec import make_plot
    from ...hspec import read_spectra
    from ...hspec import run_fit
    from ...hspec import specsource
    from ...hspec import wstat


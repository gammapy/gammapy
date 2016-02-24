# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...utils.testing import requires_data, run_cli
from ..catalog_query import cli


@pytest.mark.parametrize('args', [
    (),
    (['catalogs']),
    (['sources', '2fhl']),
    (['table-info', '2fhl']),
    # (['table-web', '2fhl']), # We don't want the tests to open up a web browser ...
    # (['--catalog', '3fgl', '--source', '3FGL J0349.9-2102', '--querytype', 'info']),
    # TODO: these are currently broken ... fix and re-activate test!
    # (['--catalog', '3fgl', '--source', '3FGL J0349.9-2102', '--querytype', 'spectrum']),
    # (['--catalog', '3fgl', '--source', '3FGL J0349.9-2102', '--querytype', 'lightcurve']),
])
@requires_data('gammapy-extra')
def test_catalog_query_main(args):
    """This test just exercises the code, the output is not checked."""
    run_cli(cli, args)

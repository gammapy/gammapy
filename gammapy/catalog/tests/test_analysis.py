# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from ...catalog import FluxDistribution


def test_FluxDistribution():
    table = Table([dict(S=42)])
    flux_distribution = FluxDistribution(table, label='dummy')
    # TODO: add useful test and assertion

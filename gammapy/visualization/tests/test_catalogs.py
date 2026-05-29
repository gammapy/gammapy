# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Licensed under a 3-clause BSD style license - see LICENSE.rst


import pytest
from numpy.testing import assert_allclose
from gammapy.catalog import SourceCatalog3PC
from gammapy.utils.testing import requires_data
from gammapy.visualization.catalogs import plot_pulse_profile_3PC


@requires_data()
def test_plot_pulse_profile_3pc_invalid_nperiod():
    """Test that invalid n_period raises ValueError."""

    catalog = SourceCatalog3PC()
    source = catalog[0]

    with pytest.raises(ValueError):
        plot_pulse_profile_3PC(source, n_period=3)


@requires_data()
def test_plot_pulse_profile_3pc_basic():
    """Test basic execution with a real 3PC source."""

    catalog = SourceCatalog3PC()

    source = catalog["J0534+2200"]
    # for reference
    # https://fermi.gsfc.nasa.gov/ssc/data/access/lat/3rd_PSR_catalog/3PC_HTML/J0534+2200.html

    axes = plot_pulse_profile_3PC(source, n_period=2)

    assert len(axes) == 6

    for ax in axes:
        xmin, xmax = ax.get_xlim()
        assert_allclose([xmin, xmax], [0, 2])


@requires_data()
def test_plot_pulse_profile_3pc_single_period():
    """Test plotting with n_period=1."""

    catalog = SourceCatalog3PC()
    source = catalog[0]

    axes = plot_pulse_profile_3PC(source, n_period=1)

    for ax in axes:
        xmin, xmax = ax.get_xlim()
        assert_allclose([xmin, xmax], [0, 1])

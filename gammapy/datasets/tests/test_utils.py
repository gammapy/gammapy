# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import MapDataset
from gammapy.datasets.utils import apply_edisp, split_dataset
from gammapy.irf import EDispKernel
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import (
    Models,
    PowerLawNormSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)


@pytest.fixture
def region_map_true():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy_true")
    m = Map.create(
        region="icrs;circle(83.63, 21.51, 1)",
        map_type="region",
        axes=[axis],
        unit="1/TeV",
    )
    m.data = np.arange(m.data.size, dtype=float).reshape(m.geom.data_shape)
    return m


def test_apply_edisp(region_map_true):
    e_true = region_map_true.geom.axes[0]
    e_reco = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)

    edisp = EDispKernel.from_diagonal_response(
        energy_axis_true=e_true, energy_axis=e_reco
    )

    m = apply_edisp(region_map_true, edisp)
    assert m.geom.data_shape == (3, 1, 1)

    e_reco = m.geom.axes[0].edges
    assert e_reco.unit == "TeV"
    assert m.geom.axes[0].name == "energy"
    assert_allclose(e_reco[[0, -1]].value, [1, 10])


def test_dataset_split():

    template_diffuse = TemplateSpatialModel.read(
        filename="/Users/qremy/Work/GitHub/gammapy-data/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz",
        normalize=False,
    )

    diffuse_iem = SkyModel(
        spectral_model=PowerLawNormSpectralModel(),
        spatial_model=template_diffuse,
        name="diffuse-iem",
    )

    dataset = MapDataset.read(
        "/Users/qremy/Work/GitHub/gammapy-data/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz"
    )
    dataset.models = Models([diffuse_iem])

    width = 4 * u.deg
    margin = 2 * u.deg
    datasets = split_dataset(dataset, width, margin)
    assert len(datasets) == 15
    assert len(datasets.models) == 1

    datasets = split_dataset(
        dataset, width=4 * u.deg, margin=2 * u.deg, split_templates=True
    )
    assert len(datasets.models) == len(datasets)
    assert len(datasets.parameters.free_parameters) == 1

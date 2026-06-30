# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for RegularizedFluxPointsEstimator
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from gammapy.datasets import MapDataset
from gammapy.estimators import FluxPoints, RegularizedFluxPointsEstimator
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    Models,
    PowerLawSpectralModel,
    SkyModel,
)


@pytest.fixture
def simple_dataset():
    energy_axis = MapAxis.from_edges(
        [0.3, 1, 3, 10] * u.TeV, name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.1,
        width=(2, 2),
        axes=[energy_axis],
    )

    counts = Map.from_geom(geom, data=10)
    exposure = Map.from_geom(geom.as_energy_true, data=1e10, unit="cm2 s")
    background = Map.from_geom(geom, data=1)

    model = SkyModel(
        spectral_model=PowerLawSpectralModel(
            index=2.0,
            amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1 * u.TeV,
        ),
        name="test-source",
    )

    dataset = MapDataset(
        counts=counts,
        exposure=exposure,
        background=background,
        models=Models([model]),
        name="test",
    )

    return dataset, model


@pytest.fixture
def energy_nodes():
    energy_nodes = [1.0, 3.0] * u.TeV
    return energy_nodes


def test_run_returns_flux_points(simple_dataset, energy_nodes):
    dataset, model = simple_dataset

    estimator = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="unpenalized",
    )

    result = estimator.run([dataset])

    assert "flux_points" in result
    assert model.name in result["flux_points"]

    flux_points = result["flux_points"][model.name]
    assert isinstance(flux_points, FluxPoints)


def test_flux_points_norm_shape(simple_dataset, energy_nodes):
    dataset, model = simple_dataset

    estimator = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="unpenalized",
    )

    result = estimator.run([dataset])
    fp = result["flux_points"][model.name]

    norms = fp.norm.data
    assert norms.shape == (len(energy_nodes), 1, 1)


def test_l2_penalty_affects_stat(simple_dataset, energy_nodes):
    dataset, model = simple_dataset

    estimator_unpen = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="unpenalized",
    )

    estimator_l2 = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="L2",
        lambda_=10,
    )

    result_unpen = estimator_unpen.run([dataset])
    result_l2 = estimator_l2.run([dataset])

    assert result_l2["stat_sum_penalty"] > 0
    assert_allclose(result_unpen["stat_sum_penalty"], 0.0)


def test_optional_errors(simple_dataset, energy_nodes):
    dataset, model = simple_dataset

    estimator = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="unpenalized",
        selection_optional=["errn-errp"],
    )

    result = estimator.run([dataset])
    fp = result["flux_points"][model.name]

    assert "norm_errn" in fp.available_quantities
    assert "norm_errp" in fp.available_quantities


def test_optional_ul(simple_dataset, energy_nodes):
    dataset, model = simple_dataset

    estimator = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="unpenalized",
        selection_optional=["ul"],
    )

    result = estimator.run([dataset])
    fp = result["flux_points"][model.name]

    assert "norm_ul" in fp.available_quantities


def test_penalty_changes_flux_points(simple_dataset, energy_nodes):
    dataset, model = simple_dataset

    estimator_unpen = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="unpenalized",
    )

    estimator_l2 = RegularizedFluxPointsEstimator(
        energy_nodes=energy_nodes,
        models=Models([model]),
        penalty_name="L2",
        lambda_=10.0,
    )

    result_unpen = estimator_unpen.run([dataset])
    result_l2 = estimator_l2.run([dataset])

    fp_unpen = result_unpen["flux_points"][model.name]
    fp_l2 = result_l2["flux_points"][model.name]

    norms_unpen = fp_unpen.norm.data
    norms_l2 = fp_l2.norm.data

    assert not np.allclose(norms_unpen, norms_l2)

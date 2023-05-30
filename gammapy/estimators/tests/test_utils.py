# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.datasets import MapDataset
from gammapy.estimators import ExcessMapEstimator
from gammapy.estimators.utils import (
    find_peaks,
    find_peaks_in_flux_map,
    resample_energy_edges,
)
from gammapy.maps import Map, MapAxis


class TestFindPeaks:
    def test_simple(self):
        """Test a simple example"""
        image = Map.create(npix=(10, 5), unit="s")
        image.data[3, 3] = 11
        image.data[3, 4] = 10
        image.data[3, 5] = 12
        image.data[3, 6] = np.nan
        image.data[0, 9] = 1e20

        table = find_peaks(image, threshold=3)

        assert len(table) == 3
        assert table["value"].unit == "s"
        assert table["ra"].unit == "deg"
        assert table["dec"].unit == "deg"

        row = table[0]
        assert tuple((row["x"], row["y"])) == (9, 0)
        assert_allclose(row["value"], 1e20)
        assert_allclose(row["ra"], 359.55)
        assert_allclose(row["dec"], -0.2)

        row = table[1]
        assert tuple((row["x"], row["y"])) == (5, 3)
        assert_allclose(row["value"], 12)

    def test_no_peak(self):
        image = Map.create(npix=(10, 5))
        image.data[3, 5] = 12

        table = find_peaks(image, threshold=12.1)
        assert len(table) == 0

    def test_constant(self):
        image = Map.create(npix=(10, 5))

        table = find_peaks(image, threshold=3)
        assert len(table) == 0

    def test_flat_map(self):
        """Test a simple example"""
        axis1 = MapAxis.from_edges([1, 2], name="axis1")
        axis2 = MapAxis.from_edges([9, 10], name="axis2")
        image = Map.create(npix=(10, 5), unit="s", axes=[axis1, axis2])
        image.data[..., 3, 3] = 11
        image.data[..., 3, 4] = 10
        image.data[..., 3, 5] = 12
        image.data[..., 3, 6] = np.nan
        image.data[..., 0, 9] = 1e20

        table = find_peaks(image, threshold=3)
        row = table[0]

        assert len(table) == 3
        assert_allclose(row["value"], 1e20)
        assert_allclose(row["ra"], 359.55)
        assert_allclose(row["dec"], -0.2)


class TestFindFluxPeaks:
    """Tests for find_peaks_in_flux_map"""

    def test_find_peaks_in_flux_map(self):
        """Test a simple example"""
        dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
        estimator = ExcessMapEstimator(
            correlation_radius="0.1 deg", energy_edges=[0.1, 10] * u.TeV
        )
        maps = estimator.run(dataset)
        table = find_peaks_in_flux_map(maps, threshold=5, min_distance=0.1 * u.deg)

        assert table["ra"].unit == "deg"
        assert table["dec"].unit == "deg"

    def test_no_peak(self):
        dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
        position = SkyCoord(1.2, 1, frame="galactic", unit="deg")
        dataset_cutout = dataset.cutout(
            position=position, width=(0.2 * u.deg, 0.2 * u.deg)
        )

        estimator = ExcessMapEstimator(
            correlation_radius="0.1 deg", energy_edges=[0.1, 10] * u.TeV
        )
        maps = estimator.run(dataset_cutout)

        table = find_peaks_in_flux_map(maps, threshold=5, min_distance=0.1 * u.deg)

        assert len(table) == 0


def test_resample_energy_edges(spectrum_dataset):
    resampled_energy_edges = resample_energy_edges(spectrum_dataset, conditions={})
    assert (resampled_energy_edges == spectrum_dataset._geom.axes["energy"].edges).all()

    with pytest.raises(ValueError):
        resample_energy_edges(
            spectrum_dataset,
            conditions={"counts_min": spectrum_dataset.counts.data.sum() + 1},
        )

    resampled_energy_edges = resample_energy_edges(
        spectrum_dataset,
        conditions={"excess_min": spectrum_dataset.excess.data[-1] + 1},
    )
    grouped = spectrum_dataset.resample_energy_axis(
        MapAxis.from_edges(edges=resampled_energy_edges, name="energy")
    )

    assert grouped.counts.data.shape == (29, 1, 1)
    assert_allclose(np.squeeze(grouped.counts)[-1], 2518.0)
    assert_allclose(np.squeeze(grouped.background)[-1], 200)

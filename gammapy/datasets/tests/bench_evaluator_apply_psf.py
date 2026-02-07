#!/usr/bin/env python
"""
Benchmark MapEvaluator.apply_psf in two fixed modes (one run prints both):

A) cube: npred is (E, Y, X)
B) image_broadcast: npred is 2D (Y, X), PSF kernel is (Ek, Ky, Kx) -> triggers broadcast path

- Fixed parameters are defined at the top
- Excludes input construction from timing
- CUDA synchronize enabled for accurate GPU timings
"""

from __future__ import annotations

import importlib
import statistics
import time

import numpy as np


# =========================
# Fixed benchmark parameters
# =========================
RUNS = 50
WARMUP = 5

WIDTH_DEG = 10.0  # spatial width in deg
BINSZ_DEG = 0.01  # pixel size in deg (smaller => larger image)
N_E = 32  # energy planes for cube mode
E_K = 32  # kernel energy planes for image_broadcast mode
PSF_SIGMA_DEG = 0.05  # PSF sigma in deg


def _cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _time_block(fn, runs=RUNS, warmup=WARMUP):
    # warmup
    for _ in range(warmup):
        _cuda_sync()
        fn()
        _cuda_sync()

    times = []
    for _ in range(runs):
        _cuda_sync()
        t0 = time.perf_counter()
        fn()
        _cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean = statistics.mean(times)
    stdev = statistics.pstdev(times) if len(times) > 1 else 0.0
    return mean, stdev


def _make_common_objects():
    TestEvaluator = importlib.import_module("gammapy.datasets.tests.test_evaluator")

    SkyCoord = TestEvaluator.SkyCoord
    MapAxis = TestEvaluator.MapAxis
    WcsGeom = TestEvaluator.WcsGeom
    PowerLawSpectralModel = TestEvaluator.PowerLawSpectralModel
    PointSpatialModel = TestEvaluator.PointSpatialModel
    SkyModel = TestEvaluator.SkyModel
    Map = TestEvaluator.Map
    PSFKernel = TestEvaluator.PSFKernel
    MapEvaluator = TestEvaluator.MapEvaluator

    import astropy.units as u

    center = SkyCoord("0 deg", "0 deg", frame="galactic")

    # reco energy axis for cube mode
    energy_axis = MapAxis.from_energy_bounds(
        "0.3 TeV", "30 TeV", nbin=N_E, name="energy"
    )
    geom = WcsGeom.create(
        skydir=center,
        width=WIDTH_DEG * u.deg,
        axes=[energy_axis],
        frame="galactic",
        binsz=BINSZ_DEG * u.deg,
    )

    spectral_model = PowerLawSpectralModel(index=2, amplitude="1e-11 TeV-1 s-1 m-2")
    spatial_model = PointSpatialModel(
        lon_0=0 * u.deg, lat_0=0 * u.deg, frame="galactic"
    )
    model = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model)

    exposure = Map.from_geom(geom.as_energy_true, unit="m2 s")
    exposure.data += 1.0

    psf = PSFKernel.from_gauss(geom, sigma=PSF_SIGMA_DEG * u.deg)

    evaluator = MapEvaluator(model=model, exposure=exposure, psf=psf)

    return TestEvaluator, evaluator, center, geom


def _bench_cube(evaluator):
    # Precompute cube npred (not timed)
    npred = evaluator.compute_npred()
    kshape = evaluator.psf.psf_kernel_map.data.shape
    xshape = npred.data.shape

    mean, stdev = _time_block(lambda: evaluator.apply_psf(npred))
    return xshape, kshape, mean, stdev


def _bench_image_broadcast(TestEvaluator, evaluator, center, geom):
    # Build 2D image npred (Y,X) (not timed)
    geom_img = geom.to_image()
    npred_img = TestEvaluator.Map.from_geom(geom_img, unit="")

    # Put a non-trivial smooth blob (float32)
    y, x = np.indices(npred_img.data.shape)
    cy, cx = (npred_img.data.shape[0] - 1) / 2.0, (npred_img.data.shape[1] - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    npred_img.data = np.exp(
        -r2 / (2.0 * (0.12 * min(npred_img.data.shape)) ** 2)
    ).astype(np.float32)

    # Replace PSF with an energy-dependent kernel (Ek,Ky,Kx)
    import astropy.units as u

    energy_axis_k = TestEvaluator.MapAxis.from_energy_bounds(
        "0.3 TeV", "30 TeV", nbin=E_K, name="energy"
    )
    geom_k = TestEvaluator.WcsGeom.create(
        skydir=center,
        width=WIDTH_DEG * u.deg,
        axes=[energy_axis_k],
        frame="galactic",
        binsz=BINSZ_DEG * u.deg,
    )
    evaluator.psf = TestEvaluator.PSFKernel.from_gauss(
        geom_k, sigma=PSF_SIGMA_DEG * u.deg
    )

    kshape = evaluator.psf.psf_kernel_map.data.shape
    xshape = npred_img.data.shape

    mean, stdev = _time_block(lambda: evaluator.apply_psf(npred_img))
    return xshape, kshape, mean, stdev


def main():
    TestEvaluator, evaluator, center, geom = _make_common_objects()

    print("=== Fixed benchmark settings ===")
    print(f"RUNS={RUNS}, WARMUP={WARMUP}")
    print(f"WIDTH_DEG={WIDTH_DEG}, BINSZ_DEG={BINSZ_DEG}")
    print(f"N_E={N_E}, E_K={E_K}, PSF_SIGMA_DEG={PSF_SIGMA_DEG}")

    # Mode A: cube
    xshape_a, kshape_a, mean_a, stdev_a = _bench_cube(evaluator)

    # Re-create evaluator fresh for mode B to avoid any cache side effects
    TestEvaluator, evaluator_b, center, geom = _make_common_objects()
    xshape_b, kshape_b, mean_b, stdev_b = _bench_image_broadcast(
        TestEvaluator, evaluator_b, center, geom
    )

    print("\n=== Results (apply_psf only) ===")

    print("\n[Mode A] cube npred (E,Y,X)")
    print("  npred.shape :", xshape_a)
    print("  kernel.shape:", kshape_a)
    print(f"  mean  = {mean_a * 1e3:.3f} ms")
    print(f"  stdev = {stdev_a * 1e3:.3f} ms")

    print("\n[Mode B] image_broadcast npred (Y,X) + kernel (Ek,Ky,Kx)")
    print("  npred.shape :", xshape_b)
    print("  kernel.shape:", kshape_b)
    print(f"  mean  = {mean_b * 1e3:.3f} ms")
    print(f"  stdev = {stdev_b * 1e3:.3f} ms")

    try:
        import torch

        print("\nTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception:
        print("\nTorch: not available")


if __name__ == "__main__":
    main()

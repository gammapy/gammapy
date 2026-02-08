#!/usr/bin/env python
"""
Benchmark MapEvaluator.apply_psf in two fixed modes, on BOTH CPU and GPU (if available),
and compare final outputs CPU vs GPU.

Mode A (cube): npred is (E, Y, X)
Mode B (image_broadcast): npred is (Y, X), PSF kernel is (Ek, Ky, Kx)

- Fixed parameters at top
- Excludes input construction from timing
- CUDA synchronize enabled for accurate GPU timings
- Compares CPU vs GPU outputs with robust metrics (single compare only):
  * max_abs, rms_abs
  * max_rel_masked: max(|a-b|/|a|) only where |a| > rel_threshold
  * allclose(rtol, atol)

Additionally:
- Sweep kernel size from 11x11 to 111x111 (odd sizes), report CPU/GPU speedup vs kernel size
"""

from __future__ import annotations

import importlib
import statistics
import time
from dataclasses import dataclass

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

# =========================
# Output-compare parameters
# =========================
COMPARE_DTYPE = np.float64  # only for comparison printing (not for timing)
RTOL = 1e-5
ATOL = 1e-6
REL_THRESHOLD = 1e-4  # for masked relative error (|CPU| > REL_THRESHOLD)

# =========================
# Kernel sweep parameters
# =========================
KERNEL_SIZES = list(range(11, 112, 10))  # 11,21,...,111

# Keep sweep shorter than the main benchmark to avoid huge runtime
RUNS_SWEEP = 10
WARMUP_SWEEP = 2


def _bytes_to_mib(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def _get_rss_bytes() -> int:
    """Process RSS in bytes (CPU RAM). Returns 0 if psutil not available."""
    try:
        import psutil
        import os

        p = psutil.Process(os.getpid())
        return int(p.memory_info().rss)
    except Exception:
        return 0


def _gpu_mem_reset_peak():
    """Reset torch CUDA peak memory stats if possible."""
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return
    # Ensure stats are reset for a clean peak measurement
    torch.cuda.reset_peak_memory_stats()
    # Optional: helps reflect allocator state changes in some setups
    _cuda_sync()


def _gpu_mem_peaks():
    """Return (max_allocated_bytes, max_reserved_bytes). (0,0) if not available."""
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return (0, 0)
    _cuda_sync()
    return (
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
    )


def _get_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _cuda_sync():
    torch = _get_torch()
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_block(fn, runs, warmup):
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


def _to_numpy(x):
    """Convert gammapy Map or array/tensor to numpy array on CPU for comparison."""
    if hasattr(x, "data"):  # gammapy Map
        x = x.data

    torch = _get_torch()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


@dataclass
class RunResult:
    mean_s: float
    stdev_s: float
    out_np: np.ndarray
    out_shape: tuple
    out_dtype: str

    # memory (per correctness run, not timed)
    rss_before: int = 0
    rss_after: int = 0
    gpu_max_allocated: int = 0
    gpu_max_reserved: int = 0


def _compare_arrays(a_np: np.ndarray, b_np: np.ndarray, *, compare_dtype=COMPARE_DTYPE):
    """Single compare report: max_abs, rms_abs, max_rel_masked, allclose, plus scale info."""
    if a_np.shape != b_np.shape:
        return {
            "shape_match": False,
            "a_shape": a_np.shape,
            "b_shape": b_np.shape,
        }

    a = a_np.astype(compare_dtype, copy=False)
    b = b_np.astype(compare_dtype, copy=False)

    diff = a - b
    abs_diff = np.abs(diff)

    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    rms_abs = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0

    # masked relative error: only where |a| is "large enough" (physically meaningful)
    mask = np.abs(a) > REL_THRESHOLD
    fraction_masked = float(np.mean(mask)) if mask.size else 0.0
    if np.any(mask):
        rel_masked = abs_diff[mask] / np.abs(a[mask])
        max_rel_masked = float(rel_masked.max()) if rel_masked.size else 0.0
    else:
        max_rel_masked = 0.0

    allclose = bool(np.allclose(a, b, rtol=RTOL, atol=ATOL))

    # "global_scale" (max abs value) only used to give context in the printout
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 0.0)
    global_scale = float(scale.max()) if scale.size else 0.0
    eps_used = max(global_scale * 1e-12, 1e-30)

    return {
        "shape_match": True,
        "compare_dtype": str(np.dtype(compare_dtype)),
        "max_abs": max_abs,
        "rms_abs": rms_abs,
        "max_rel_masked": max_rel_masked,
        "fraction_masked": fraction_masked,
        "allclose": allclose,
        "rtol": RTOL,
        "atol": ATOL,
        "rel_threshold": REL_THRESHOLD,
        "global_scale": global_scale,
        "eps_used": eps_used,
    }


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


def _set_psf_kernel_size(TestEvaluator, evaluator, geom, *, kernel_size: int):
    """
    Force PSF kernel spatial shape to (kernel_size, kernel_size) using max_radius.

    PSFKernel.from_gauss supports max_radius (Angle) to set desired kernel map size. :contentReference[oaicite:1]{index=1}
    """
    import astropy.units as u

    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd.")

    radius_pix = (kernel_size - 1) / 2.0
    max_radius = (radius_pix * BINSZ_DEG) * u.deg

    evaluator.psf = TestEvaluator.PSFKernel.from_gauss(
        geom, sigma=PSF_SIGMA_DEG * u.deg, max_radius=max_radius
    )

    # sanity: actual kernel may differ by 1 depending on internal rounding; we’ll report actual
    ky, kx = evaluator.psf.psf_kernel_map.data.shape[-2:]
    return (ky, kx)


def _prepare_mode_a_cube(evaluator):
    return evaluator.compute_npred()


def _prepare_mode_b_image_broadcast(TestEvaluator, evaluator, center, geom):
    geom_img = geom.to_image()
    npred_img = TestEvaluator.Map.from_geom(geom_img, unit="")

    y, x = np.indices(npred_img.data.shape)
    cy, cx = (npred_img.data.shape[0] - 1) / 2.0, (npred_img.data.shape[1] - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    npred_img.data = np.exp(
        -r2 / (2.0 * (0.12 * min(npred_img.data.shape)) ** 2)
    ).astype(np.float32)

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
    return npred_img, geom_k


def _run_apply_psf_timed(evaluator, npred, *, force_cpu: bool, runs: int, warmup: int):
    """
    Run one correctness evaluation (not timed), measure memory around it,
    then time apply_psf only.

    Memory measurement:
      - CPU: process RSS before/after correctness run
      - GPU: torch.cuda peak allocated/reserved during correctness run (if GPU path)
    """
    torch = _get_torch()
    has_cuda = (torch is not None) and torch.cuda.is_available()

    rss0 = _get_rss_bytes()
    gpu_max_alloc = 0
    gpu_max_resv = 0

    # correctness output (single run, outside timing)
    if force_cpu:
        # CPU path: measure RSS delta
        out0 = evaluator.apply_psf(npred, force_cpu=True)
        rss1 = _get_rss_bytes()
    else:
        # GPU path: measure GPU VRAM peaks (PyTorch allocator) during this call
        if has_cuda:
            _gpu_mem_reset_peak()
        out0 = evaluator.apply_psf(npred)
        if has_cuda:
            gpu_max_alloc, gpu_max_resv = _gpu_mem_peaks()
        rss1 = _get_rss_bytes()

    out0_np = _to_numpy(out0)

    # timing (apply_psf only)
    if force_cpu:
        mean, stdev = _time_block(
            lambda: evaluator.apply_psf(npred, force_cpu=True),
            runs=runs,
            warmup=warmup,
        )
    else:
        mean, stdev = _time_block(
            lambda: evaluator.apply_psf(npred),
            runs=runs,
            warmup=warmup,
        )

    return RunResult(
        mean_s=mean,
        stdev_s=stdev,
        out_np=out0_np,
        out_shape=out0_np.shape,
        out_dtype=str(out0_np.dtype),
        rss_before=rss0,
        rss_after=rss1,
        gpu_max_allocated=gpu_max_alloc,
        gpu_max_reserved=gpu_max_resv,
    )


def _print_header():
    print("=== Fixed benchmark settings ===")
    print(f"RUNS={RUNS}, WARMUP={WARMUP}")
    print(f"WIDTH_DEG={WIDTH_DEG}, BINSZ_DEG={BINSZ_DEG}")
    print(f"N_E={N_E}, E_K={E_K}, PSF_SIGMA_DEG={PSF_SIGMA_DEG}")

    print("\n=== Compare settings ===")
    print(f"COMPARE_DTYPE={np.dtype(COMPARE_DTYPE)}")
    print(f"RTOL={RTOL}, ATOL={ATOL}")
    print(f"REL_THRESHOLD={REL_THRESHOLD}")

    print("\n=== Kernel sweep settings ===")
    print(f"KERNEL_SIZES={KERNEL_SIZES}")
    print(f"RUNS_SWEEP={RUNS_SWEEP}, WARMUP_SWEEP={WARMUP_SWEEP}")

    torch = _get_torch()
    if torch is None:
        print("\nTorch: not available")
    else:
        print(f"\nTorch: {torch.__version__}")
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))


def _report_mode(mode_name, npred_shape, kernel_shape, cpu_res, gpu_res):
    print(f"\n=== {mode_name} ===")
    print("  npred.shape :", npred_shape)
    print("  kernel.shape:", kernel_shape)

    print("\n  [CPU]")
    print(f"    mean  = {cpu_res.mean_s * 1e3:.3f} ms")
    print(f"    stdev = {cpu_res.stdev_s * 1e3:.3f} ms")
    print(f"    out.shape={cpu_res.out_shape} dtype={cpu_res.out_dtype}")
    if cpu_res.rss_before and cpu_res.rss_after:
        print(
            f"    RSS (CPU)    : "
            f"{_bytes_to_mib(cpu_res.rss_before):.1f} -> {_bytes_to_mib(cpu_res.rss_after):.1f} MiB "
            f"(delta={_bytes_to_mib(cpu_res.rss_after - cpu_res.rss_before):+.1f} MiB)"
        )
    else:
        print("    RSS (CPU)    : (psutil not available)")

    if gpu_res is None:
        print("\n  [GPU] (skipped: CUDA not available)")
        return

    print("\n  [GPU]")
    print(f"    mean  = {gpu_res.mean_s * 1e3:.3f} ms")
    print(f"    stdev = {gpu_res.stdev_s * 1e3:.3f} ms")
    print(f"    out.shape={gpu_res.out_shape} dtype={gpu_res.out_dtype}")
    if gpu_res.gpu_max_allocated or gpu_res.gpu_max_reserved:
        print(
            f"    VRAM peaks   : "
            f"allocated={_bytes_to_mib(gpu_res.gpu_max_allocated):.1f} MiB, "
            f"reserved={_bytes_to_mib(gpu_res.gpu_max_reserved):.1f} MiB"
        )
    else:
        print("    VRAM peaks   : (torch/cuda not available or peak stats unavailable)")

    cmp_stats = _compare_arrays(
        cpu_res.out_np, gpu_res.out_np, compare_dtype=COMPARE_DTYPE
    )
    print("\n  [CPU vs GPU output diff]")
    if not cmp_stats.get("shape_match", False):
        print("    shape mismatch:", cmp_stats)
    else:
        print(f"    compare_dtype  = {cmp_stats['compare_dtype']}")
        print(f"    max_abs        = {cmp_stats['max_abs']:.6e}")
        print(f"    rms_abs        = {cmp_stats['rms_abs']:.6e}")
        print(
            f"    max_rel_masked = {cmp_stats['max_rel_masked']:.6e} "
            f"(mask: |CPU|>{cmp_stats['rel_threshold']}, fraction={cmp_stats['fraction_masked']:.6f})"
        )
        print(
            f"    allclose       = {cmp_stats['allclose']} (rtol={cmp_stats['rtol']}, atol={cmp_stats['atol']})"
        )
        print(
            f"    global_scale   = {cmp_stats['global_scale']:.6e}, eps_used={cmp_stats['eps_used']:.6e}"
        )


def _print_sweep_table(title, rows):
    """
    rows: list of dict with keys:
      kernel_req, kernel_actual, cpu_ms, gpu_ms, speedup
    """
    print(f"\n=== Kernel sweep: {title} (CPU vs GPU speedup) ===")
    print(
        f"{'kernel_req':>10}  {'kernel_act':>10}  {'CPU ms':>10}  {'GPU ms':>10}  {'speedup':>10}"
    )
    for r in rows:
        print(
            f"{r['kernel_req']:>10}  {r['kernel_actual']:>10}  "
            f"{r['cpu_ms']:>10.3f}  {r['gpu_ms']:>10.3f}  {r['speedup']:>10.2f}"
        )


def main():
    _print_header()

    torch = _get_torch()
    has_cuda = (torch is not None) and torch.cuda.is_available()

    # ---------------------
    # Mode A: cube (E,Y,X)
    # ---------------------
    TestEvaluator, evaluator_a, center, geom = _make_common_objects()
    npred_a = _prepare_mode_a_cube(evaluator_a)

    kshape_a = evaluator_a.psf.psf_kernel_map.data.shape
    xshape_a = _to_numpy(npred_a).shape

    cpu_a = _run_apply_psf_timed(
        evaluator_a, npred_a, force_cpu=True, runs=RUNS, warmup=WARMUP
    )
    gpu_a = (
        _run_apply_psf_timed(
            evaluator_a, npred_a, force_cpu=False, runs=RUNS, warmup=WARMUP
        )
        if has_cuda
        else None
    )

    _report_mode(
        mode_name="[Mode A] cube npred (E,Y,X)",
        npred_shape=xshape_a,
        kernel_shape=kshape_a,
        cpu_res=cpu_a,
        gpu_res=gpu_a,
    )

    # ------------------------------------------------
    # Mode B: image_broadcast npred (Y,X) + kernel (Ek,Ky,Kx)
    # ------------------------------------------------
    TestEvaluator, evaluator_b, center, geom = _make_common_objects()
    npred_b, geom_k_b = _prepare_mode_b_image_broadcast(
        TestEvaluator, evaluator_b, center, geom
    )

    kshape_b = evaluator_b.psf.psf_kernel_map.data.shape
    xshape_b = _to_numpy(npred_b).shape

    cpu_b = _run_apply_psf_timed(
        evaluator_b, npred_b, force_cpu=True, runs=RUNS, warmup=WARMUP
    )
    gpu_b = (
        _run_apply_psf_timed(
            evaluator_b, npred_b, force_cpu=False, runs=RUNS, warmup=WARMUP
        )
        if has_cuda
        else None
    )

    _report_mode(
        mode_name="[Mode B] image_broadcast npred (Y,X) + kernel (Ek,Ky,Kx)",
        npred_shape=xshape_b,
        kernel_shape=kshape_b,
        cpu_res=cpu_b,
        gpu_res=gpu_b,
    )

    # ==========================
    # Kernel size sweep
    # ==========================
    if not has_cuda:
        print("\nCUDA not available -> skipping kernel sweep (needs GPU for speedup).")
        return

    # ---- Sweep for Mode A ----
    sweep_rows_a = []
    # fresh evaluator to avoid any caching across sizes
    TestEvaluator, evaluator_sweep_a, center, geom = _make_common_objects()
    npred_sweep_a = _prepare_mode_a_cube(evaluator_sweep_a)

    for kreq in KERNEL_SIZES:
        ky, kx = _set_psf_kernel_size(
            TestEvaluator, evaluator_sweep_a, geom, kernel_size=kreq
        )
        # timing only; we don't do output compare per size (you asked to keep only one compare)
        cpu = _run_apply_psf_timed(
            evaluator_sweep_a,
            npred_sweep_a,
            force_cpu=True,
            runs=RUNS_SWEEP,
            warmup=WARMUP_SWEEP,
        )
        gpu = _run_apply_psf_timed(
            evaluator_sweep_a,
            npred_sweep_a,
            force_cpu=False,
            runs=RUNS_SWEEP,
            warmup=WARMUP_SWEEP,
        )
        speedup = cpu.mean_s / gpu.mean_s if gpu.mean_s > 0 else float("inf")
        sweep_rows_a.append(
            {
                "kernel_req": f"{kreq}x{kreq}",
                "kernel_actual": f"{ky}x{kx}",
                "cpu_ms": cpu.mean_s * 1e3,
                "gpu_ms": gpu.mean_s * 1e3,
                "speedup": speedup,
            }
        )

    _print_sweep_table("Mode A (cube)", sweep_rows_a)

    # ---- Sweep for Mode B ----
    sweep_rows_b = []
    TestEvaluator, evaluator_sweep_b, center, geom = _make_common_objects()
    npred_sweep_b, geom_k_sweep = _prepare_mode_b_image_broadcast(
        TestEvaluator, evaluator_sweep_b, center, geom
    )

    for kreq in KERNEL_SIZES:
        ky, kx = _set_psf_kernel_size(
            TestEvaluator, evaluator_sweep_b, geom_k_sweep, kernel_size=kreq
        )
        cpu = _run_apply_psf_timed(
            evaluator_sweep_b,
            npred_sweep_b,
            force_cpu=True,
            runs=RUNS_SWEEP,
            warmup=WARMUP_SWEEP,
        )
        gpu = _run_apply_psf_timed(
            evaluator_sweep_b,
            npred_sweep_b,
            force_cpu=False,
            runs=RUNS_SWEEP,
            warmup=WARMUP_SWEEP,
        )
        speedup = cpu.mean_s / gpu.mean_s if gpu.mean_s > 0 else float("inf")
        sweep_rows_b.append(
            {
                "kernel_req": f"{kreq}x{kreq}",
                "kernel_actual": f"{ky}x{kx}",
                "cpu_ms": cpu.mean_s * 1e3,
                "gpu_ms": gpu.mean_s * 1e3,
                "speedup": speedup,
            }
        )

    _print_sweep_table("Mode B (image_broadcast)", sweep_rows_b)


if __name__ == "__main__":
    main()

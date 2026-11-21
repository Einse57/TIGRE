"""Times TIGRE's projectors and algorithms on the backend it was built with (CUDA or SYCL).

The same script, with the same arguments, on a CUDA build and on a SYCL build gives measurements
of the same code on the same problem. Each run writes a JSON file; ``--compare`` puts two of them
side by side.

What is timed (each with NumPy arrays in and out, so the host-device copies are included):

    Ax interpolated, Ax Siddon      forward projection
    Atb FDK, Atb matched            backprojection
    fdk                             algorithms.fdk
    ossart                          algorithms.ossart, --niter iterations, blocksize 20
    cgls                            algorithms.cgls, --niter iterations
    asd_pocs                        algorithms.asd_pocs, --niter iterations (20 TV iterations each)
    minTV, AwminTV                  TV gradient descent on the image, 50 iterations
    tvdenoise                       im3ddenoise (TV proximal), 50 iterations
    CTnoise                         Poisson and Gaussian noise on the projections

Problem: TIGRE's default cone-beam geometry at n^3 voxels and an n x n detector, --angles angles
over 360 degrees, and the head phantom (tigre.utilities.sample_loader). Every operation runs once
first (on the SYCL build this first call also compiles the kernels for the device; its time is
reported as "first"), then --reps times; the report gives the median and the minimum.

Each result also has a checksum of its output (sum and L2 norm, in double), to check that both
builds computed the same problem: these should agree closely between backends, except CTnoise
(the random numbers differ by design).

Usage (outside the TIGRE source tree, so that the installed TIGRE is imported):

    python backend_times.py --out cuda.json                  # 128^3 and 256^3, 1 GPU
    python backend_times.py --sizes 256 512 --out cuda.json  # other sizes
    python backend_times.py --gpus 2 --out cuda_2gpu.json    # the first 2 GPUs TIGRE lists
    python backend_times.py --compare sycl.json cuda.json    # two results side by side

GPUs are the first --gpus of the ones TIGRE lists (tigre.utilities.gpu.GpuIds). On the SYCL build,
TIGRE_SYCL_ALL_DEVICES=1 also lists the CPU, after the GPUs.

This script is not a test: pytest does not collect it, and it is not meant for CI.
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import time

import numpy as np


def tigre_version():
    try:
        return importlib.metadata.version("tigre")
    except importlib.metadata.PackageNotFoundError:
        return "?"


def backend_info(gpuids):
    from tigre.utilities.gpu import getGpuNames

    try:
        from _gpuUtils import getBackend

        backend = getBackend()
    except ImportError:  # TIGRE before the SYCL backend: CUDA only
        backend = "cuda"
    names = getGpuNames()
    return {
        "backend": backend,
        "devices": [names[d] if d < len(names) else str(d) for d in gpuids.devices],
    }


def operations(n, nangles, niter, gpuids):
    """(name, function) for every timed operation, on the same geometry and data."""
    import tigre
    import tigre.algorithms as algs
    from tigre.utilities import CTnoise, sample_loader
    from tigre.utilities.im_3d_denoise import im3ddenoise
    from _minTV import minTV
    from _AwminTV import AwminTV

    geo = tigre.geometry(mode="cone", default=True, nVoxel=np.array([n, n, n]))
    geo.nDetector = np.array([n, n])
    geo.dDetector = np.array([0.8, 0.8]) * 512 / n
    geo.sDetector = geo.nDetector * geo.dDetector
    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False, dtype=np.float32)
    img = sample_loader.load_head_phantom(geo.nVoxel).astype(np.float32)
    proj = tigre.Ax(img, geo, angles, gpuids=gpuids)
    kw = dict(gpuids=gpuids, verbose=False)
    return [
        ("Ax interpolated", lambda: tigre.Ax(img, geo, angles, "interpolated", gpuids=gpuids)),
        ("Ax Siddon", lambda: tigre.Ax(img, geo, angles, "Siddon", gpuids=gpuids)),
        ("Atb FDK", lambda: tigre.Atb(proj, geo, angles, "FDK", gpuids=gpuids)),
        ("Atb matched", lambda: tigre.Atb(proj, geo, angles, "matched", gpuids=gpuids)),
        ("fdk", lambda: algs.fdk(proj, geo, angles, gpuids=gpuids)),
        ("ossart", lambda: algs.ossart(proj, geo, angles, niter, blocksize=20, **kw)),
        ("cgls", lambda: algs.cgls(proj, geo, angles, niter, **kw)),
        ("asd_pocs", lambda: algs.asd_pocs(proj, geo, angles, niter, tviter=20, **kw)),
        ("minTV", lambda: minTV(img, 0.01, 50, gpuids=gpuids)),
        ("AwminTV", lambda: AwminTV(img, 0.01, 50, -0.005, gpuids=gpuids)),
        ("tvdenoise", lambda: im3ddenoise(img, 50, 15.0, gpuids=gpuids)),
        ("CTnoise", lambda: CTnoise.add(proj, Poisson=1e5, Gaussian=np.array([0, 10]), seed=0)),
    ]


def checksum(result):
    if isinstance(result, tuple):  # algorithms may return (image, quality measures)
        result = result[0]
    a = np.asarray(result, dtype=np.float64)
    return {"sum": float(a.sum()), "l2": float(np.sqrt((a * a).sum()))}


def measure(n, nangles, niter, reps, gpuids, only):
    rows = []
    for name, call in operations(n, nangles, niter, gpuids):
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        out = call()
        first = time.perf_counter() - t0
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            call()
            times.append(time.perf_counter() - t0)
        row = {
            "op": name,
            "n": n,
            "first": first,
            "times": times,
            "median": statistics.median(times),
            "min": min(times),
            "check": checksum(out),
        }
        rows.append(row)
        print(
            "  %-16s n=%-5d median %9.4f s   min %9.4f s   first %9.4f s"
            % (name, n, row["median"], row["min"], first),
            flush=True,
        )
    return rows


def compare(path_a, path_b):
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    for label, r in (("A", a), ("B", b)):
        m = r["meta"]
        print(
            "%s: TIGRE %s, %s on %s, %s  (%s)"
            % (label, m["tigre"], m["backend"], ", ".join(m["devices"]), m["os"], m["when"])
        )
    rb = {(r["op"], r["n"]): r for r in b["results"]}
    print(
        "\n%-16s %6s %12s %12s %8s %12s"
        % ("operation", "n", "A median s", "B median s", "B / A", "check diff")
    )
    for ra in a["results"]:
        key = (ra["op"], ra["n"])
        if key not in rb:
            continue
        r2 = rb[key]
        l2a, l2b = ra["check"]["l2"], r2["check"]["l2"]
        diff = abs(l2a - l2b) / max(abs(l2a), 1e-30)
        print(
            "%-16s %6d %12.4f %12.4f %8.3f %12.2e%s"
            % (
                key[0],
                key[1],
                ra["median"],
                r2["median"],
                r2["median"] / ra["median"],
                diff,
                "  (random)" if key[0] == "CTnoise" else "",
            )
        )
    print("\ncheck diff: relative difference of the outputs' L2 norms")
    print("(the same problem: small, except CTnoise).")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 256],
        help="n for n^3 voxels and n x n detector",
    )
    p.add_argument("--angles", type=int, default=360)
    p.add_argument("--niter", type=int, default=5, help="iterations of ossart, cgls and asd_pocs")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--gpus", type=int, default=1, help="use the first GPUS devices TIGRE lists")
    p.add_argument("--only", nargs="+", help="time only these operations (names as in the report)")
    p.add_argument("--out", help="write the results to this JSON file")
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.json", "B.json"),
        help="print two saved results side by side",
    )
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    import tigre  # noqa: F401  (first: on Windows it adds CUDA's DLL folder)
    from tigre.utilities.gpu import GpuIds

    gpuids = GpuIds()
    if len(gpuids.devices) < args.gpus:
        sys.exit(
            "TIGRE lists %d device(s); --gpus %d asked for more" % (len(gpuids.devices), args.gpus)
        )
    gpuids.devices = gpuids.devices[: args.gpus]
    meta = backend_info(gpuids)
    meta.update(
        {
            "tigre": tigre_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "when": time.strftime("%Y-%m-%d %H:%M %z"),
            "args": vars(args),
        }
    )
    print(
        "TIGRE %s, %s backend, on %s" % (meta["tigre"], meta["backend"], ", ".join(meta["devices"]))
    )
    results = []
    for n in args.sizes:
        results += measure(n, args.angles, args.niter, args.reps, gpuids, args.only)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)
        print("wrote " + args.out)


if __name__ == "__main__":
    main()

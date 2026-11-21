"""SYCL build: minTV, AwminTV and tvdenoise with pieces that would come out empty.

The TV functions cut the image in z into deviceCount*splits pieces of
ceil(slices / pieces) slices. That can leave the last pieces empty, and an empty
piece's size wrapped around (unsigned) to a huge one. tvPiecesWithoutEmpty
(gpuUtils.hpp) now picks pieces of which none is empty (see
test_tv_few_slices_multigpu.py, which needs two or more devices).

Here one device is listed two or three times, so the multi-device paths run on any
SYCL device, including the CPU device in CI: few slices with no memory limit, and
a 64x256x256 image on two listings with an 8 MB limit (12 pieces of 6 slices for 64
slices before, the last one -2). Each case runs in a subprocess.

test_halos_stay_inside_the_image: minTV, AwminTV and minPICCS read only the caller's
arrays (#899), with pieces thinner than the halo; as test_tv_host_bounds.py, which
needs PyTorch or three GPUs, but with listings and TIGRE_SYCL_MEMORY_LIMIT_MB.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_gpuUtils = pytest.importorskip("_gpuUtils")
if getattr(_gpuUtils, "getBackend", lambda: "cuda")() != "sycl":
    pytest.skip("SYCL build only (one device listed several times)", allow_module_level=True)

SCRIPT = r"""
import json, sys
import numpy as np
from tigre.utilities.gpu import GpuIds
from _minTV import minTV
from _AwminTV import AwminTV
from _tv_proximal import tvdenoise

cases, out_path = json.loads(sys.argv[1]), sys.argv[2]
calls = {
    "minTV": lambda img, ids: minTV(img, 0.01, 10, gpuids=ids),
    "AwminTV": lambda img, ids: AwminTV(img, 0.01, 10, -0.005, gpuids=ids),
    "tvdenoise": lambda img, ids: tvdenoise(img, 10, 15.0, gpuids=ids),
}
first = GpuIds().devices[0]
out = {}
for name, shape, listings in cases:
    img = np.random.default_rng(shape[0]).random(shape, dtype=np.float32)
    ids = GpuIds()
    ids.devices = [first] * listings
    key = "%s_%d_%d" % (name, shape[0], listings)
    try:
        out[key] = calls[name](img, ids)
    except RuntimeError as err:
        out[key] = np.array(str(err))
np.savez(out_path, **out)
"""


def _run(tmp_path, cases, limit_mb=None):
    script = tmp_path / "few_slices.py"
    script.write_text(SCRIPT)
    out = tmp_path / "out.npz"
    env = dict(os.environ)
    env.pop("TIGRE_SYCL_MEMORY_LIMIT_MB", None)
    if limit_mb is not None:
        env["TIGRE_SYCL_MEMORY_LIMIT_MB"] = str(limit_mb)
    result = subprocess.run([sys.executable, str(script), json.dumps(cases), str(out)], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(np.load(out))


def _check(result, ref, name, tolerance):
    assert result.dtype == np.float32, str(result)
    assert np.isfinite(result).all()
    diff = np.abs(result - ref).max() / np.abs(ref).max()
    assert diff < tolerance, (name, diff)


@pytest.mark.parametrize("listings", [2, 3])
def test_few_slices_match_one_listing(tmp_path, listings):
    names = ("minTV", "AwminTV", "tvdenoise")
    sizes = range(1, 2 * listings + 2)
    cases = [(n, (nz, 32, 40), k) for n in names for nz in sizes for k in (1, listings)]
    out = _run(tmp_path, cases)
    for name in names:
        for nz in sizes:
            _check(out["%s_%d_%d" % (name, nz, listings)], out["%s_%d_1" % (name, nz)],
                   (name, nz), 1e-6)


def test_two_listings_with_little_memory(tmp_path):
    # 8 MB: 12 pieces of 6 slices for 64 slices before (the last one -2 slices).
    names = ("minTV", "AwminTV", "tvdenoise")
    cases = [(n, (64, 256, 256), 2) for n in names]
    split = _run(tmp_path, cases, limit_mb=8)
    ref = _run(tmp_path, [(n, (64, 256, 256), 1) for n in names])
    for name in names:
        # minTV/AwminTV: each piece is normalised by its own gradient norm (as in
        # test_sycl_low_memory.py); tvdenoise: the unsplit result.
        tolerance = 1e-6 if name == "tvdenoise" else 5e-2
        _check(split["%s_64_2" % name], ref["%s_64_1" % name], name, tolerance)


GUARD_SCRIPT = r"""
import ctypes, json, mmap, sys
import numpy as np
from tigre.utilities.gpu import GpuIds
from _minTV import minTV
from _AwminTV import AwminTV
from _minPICCS import minPICCS

libc = ctypes.CDLL(None, use_errno=True)
libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
keep = []


def guarded(values):
    # A copy of values between two pages that cannot be read or written.
    page = mmap.PAGESIZE
    nbytes = values.size * 4
    body = (nbytes + page - 1) // page * page
    buf = mmap.mmap(-1, body + 2 * page)
    base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    start = base + page + body - nbytes  # the array ends right before the second guard page
    for addr in (base, base + page + body):
        assert libc.mprotect(ctypes.c_void_p(addr), page, 0) == 0
    keep.append(buf)
    arr = np.frombuffer(buf, dtype=np.float32, count=values.size, offset=start - base).reshape(values.shape)
    arr[:] = values
    return arr


shape, listings, names = json.loads(sys.argv[1])
rng = np.random.default_rng(0)
img = guarded(rng.random(shape, dtype=np.float32))
prior = guarded(rng.random(shape, dtype=np.float32))
ids = GpuIds()
ids.devices = [ids.devices[0]] * listings
calls = {"minTV": lambda: minTV(img, 0.05, 5, gpuids=ids),
         "AwminTV": lambda: AwminTV(img, 0.05, 5, -0.005, gpuids=ids),
         "minPICCS": lambda: minPICCS(img, prior, 0.05, 5, 0.5, gpuids=ids)}
out = {name: calls[name]() for name in names}
np.savez(sys.argv[2], **out)
"""


ALL = ["minTV", "AwminTV", "minPICCS"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses mprotect for guard pages")
@pytest.mark.parametrize("shape, listings, limit_mb, names", [
    # 64 slices with a 6 MB limit: minTV and AwminTV in 22 pieces of 3, the last of 1; minPICCS (one more
    # image-sized input) in pieces of 1. With 4 MB (too little for minPICCS), minTV and AwminTV in pieces of 1.
    ((64, 256, 256), 1, 6, ALL),
    ((64, 256, 256), 1, 4, ["minTV", "AwminTV"]),
    # 7 slices on 3 listings: pieces of 3, 3 and 1.
    ((7, 32, 40), 3, None, ALL),
])
def test_halos_stay_inside_the_image(tmp_path, shape, listings, limit_mb, names):
    # Next to a piece thinner than the halo, the halo used to be copied from past the end of the caller's
    # arrays, or before their start (#899). Here the image and the prior lie between two pages that cannot
    # be read, so such a read ends the subprocess.
    script = tmp_path / "guard.py"
    script.write_text(GUARD_SCRIPT)
    out = tmp_path / "guard.npz"
    env = dict(os.environ)
    env.pop("TIGRE_SYCL_MEMORY_LIMIT_MB", None)
    if limit_mb is not None:
        env["TIGRE_SYCL_MEMORY_LIMIT_MB"] = str(limit_mb)
    result = subprocess.run([sys.executable, str(script), json.dumps([shape, listings, names]), str(out)], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (result.returncode, result.stdout + result.stderr)
    for name, res in np.load(out).items():
        assert np.isfinite(res).all(), name

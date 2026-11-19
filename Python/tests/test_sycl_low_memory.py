"""SYCL build: the TV regularisers and Ax/Atb with little device memory.

TIGRE_SYCL_MEMORY_LIMIT_MB caps the memory the SYCL build plans with, so the
image-splitting and "out of memory" paths of #864 (TV) and 1a36447 (Ax/Atb)
run on any SYCL device, including the CPU device in CI. Each case runs in a
subprocess, since the limit is read from the environment.

What a split result should be (as in test_tv_low_memory.py for the CUDA build):
- minTV/AwminTV: close to the unsplit result; each piece is normalised by its
  own gradient norm, as in the CUDA build.
- tvdenoise: only that it finishes with a finite result. Split, it differs from
  the unsplit result in the CUDA build too.
- Ax (interpolated), Atb: the unsplit result.
- Ax (Siddon): the unsplit result up to rounding, since the pieces add up in a
  different order (as in test_siddon_split.py for the CUDA build, #859). A ray
  that runs exactly in the plane between two pieces is still dropped (CUDA and
  SYCL alike): with pieces of 2 slices, the central detector row lies in one.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_gpuUtils = pytest.importorskip("_gpuUtils")
if getattr(_gpuUtils, "getBackend", lambda: "cuda")() != "sycl":
    pytest.skip("SYCL build only (TIGRE_SYCL_MEMORY_LIMIT_MB)", allow_module_level=True)

SCRIPT = r"""
import sys
import numpy as np
import tigre
from _minTV import minTV
from _AwminTV import AwminTV
from _tv_proximal import tvdenoise

img = np.random.default_rng(0).random((64, 256, 256), dtype=np.float32)  # 16 MB, 256 kB per slice
geo = tigre.geometry(mode="cone", default=True, nVoxel=np.array(img.shape))
geo.sVoxel = np.array([64.0, 256.0, 256.0])
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.nDetector = np.array([61, 67])
geo.dDetector = geo.sDetector / geo.nDetector
angles = np.linspace(0, 2 * np.pi, 8, endpoint=False, dtype=np.float32)
calls = {
    "minTV": lambda: minTV(img, 0.05, 5),
    "AwminTV": lambda: AwminTV(img, 0.05, 5, -0.005),
    "tvdenoise": lambda: tvdenoise(img, 10, 15.0),
    "Ax": lambda: tigre.Ax(img, geo, angles, "interpolated"),
    "AxSiddon": lambda: tigre.Ax(img, geo, angles, "Siddon"),
    "Atb": lambda: tigre.Atb(np.ones((len(angles), 61, 67), dtype=np.float32), geo, angles),
}
out = {}
for name, call in calls.items():
    try:
        out[name] = call()
    except RuntimeError as err:
        out[name] = np.array(str(err))
np.savez(sys.argv[1], **out)
"""


def _run(tmp_path, limit_mb):
    script = tmp_path / "low_memory.py"
    script.write_text(SCRIPT)
    out = tmp_path / "out_{}.npz".format(limit_mb)
    env = dict(os.environ)
    env.pop("TIGRE_SYCL_MEMORY_LIMIT_MB", None)
    if limit_mb is not None:
        env["TIGRE_SYCL_MEMORY_LIMIT_MB"] = str(limit_mb)
    subprocess.run([sys.executable, str(script), str(out)], env=env, check=True, timeout=600)
    return dict(np.load(out))


@pytest.fixture(scope="module")
def unlimited(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("unlimited"), None)


def _assert_close(name, result, ref, finite_only=("tvdenoise",)):
    assert result.dtype == np.float32, (name, str(result))
    assert np.isfinite(result).all(), name
    assert result.shape == ref.shape, name
    if name in finite_only:
        return  # see the module docstring
    diff = np.abs(result - ref).max() / np.abs(ref).max()
    tolerance = {"minTV": 5e-2, "AwminTV": 5e-2, "AxSiddon": 1e-4}.get(name, 1e-5)
    assert diff < tolerance, (name, diff)


def _assert_out_of_memory(name, result):
    assert result.dtype.kind == "U", (name, "did not raise")
    assert "out of memory" in str(result), (name, str(result))


def test_split_results_match_the_unsplit_ones(tmp_path, unlimited):
    # 24 MB: the TV functions (3 or 5 images) and Ax (image and projection buffers) split.
    split = _run(tmp_path, 24)
    for name, ref in unlimited.items():
        _assert_close(name, split[name], ref)


def test_many_image_pieces(tmp_path, unlimited):
    # 2 MB: Ax splits the 64 slices into 11 pieces of 6 (12 pieces of 6 would leave the
    # last one with a negative number of slices), Atb splits too. The TV functions'
    # smallest working set (about 4 MB) does not fit.
    result = _run(tmp_path, 2)
    for name in ("Ax", "AxSiddon", "Atb"):
        _assert_close(name, result[name], unlimited[name])
    for name in ("minTV", "AwminTV", "tvdenoise"):
        _assert_out_of_memory(name, result[name])


def test_too_little_memory_raises_out_of_memory(tmp_path, unlimited):
    # 1 MB: less than Atb's smallest working set too (projection buffers and one slice).
    # Ax still fits its buffers and one slice, in pieces of 2 slices; the central detector
    # row then runs in the plane between two pieces, which Siddon drops.
    result = _run(tmp_path, 1)
    for name in ("minTV", "AwminTV", "tvdenoise", "Atb"):
        _assert_out_of_memory(name, result[name])
    for name in ("Ax", "AxSiddon"):
        _assert_close(name, result[name], unlimited[name], finite_only=("AxSiddon",))

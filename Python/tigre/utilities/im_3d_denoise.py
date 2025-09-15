import numpy as np

# SYCL-only mode: Check if TV proximal extension is available
try:
    from _tv_proximal import tvdenoise
    HAS_TV_PROXIMAL_EXTENSION = True
except ImportError:
    # SYCL-only mode - use fallback implementation
    from .tv_stubs import tvdenoise
    HAS_TV_PROXIMAL_EXTENSION = False
    print("⚠️  CUDA tv_proximal extension not available - using fallback implementation")

from .gpu import GpuIds


def im3ddenoise(img, iter=50, lmbda=15.0, gpuids=None):
    imgmin = np.amin(img.ravel())
    img = img - imgmin
    imgmax = np.amax(img.ravel())
    img = img / imgmax

    if gpuids is None:
        gpuids = GpuIds()

    img = tvdenoise(img, iter, lmbda, gpuids)

    img *= imgmax
    img += imgmin

    return img

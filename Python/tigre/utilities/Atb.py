import copy

import numpy as np

# SYCL-only mode: Check if Atb extension is available, otherwise use backend
try:
    from _Atb import _Atb_ext
    HAS_ATB_EXTENSION = True
    USE_BACKEND = False
except ImportError:
    # SYCL-only mode - use backend implementation
    HAS_ATB_EXTENSION = False
    USE_BACKEND = True
    print("⚠️  CUDA Atb extension not available - using SYCL backend")

from .gpu import GpuIds


def Atb(projections, geo, angles, backprojection_type="FDK", **kwargs):
    if projections.dtype != np.float32:
        raise TypeError("Input data should be float32, not " + str(projections.dtype))
    if not np.isreal(projections).all():
        raise ValueError("Complex types not compatible for back projection.")
    geox = copy.deepcopy(geo)
    geox.check_geo(angles)
    """
    Here we cast all values in geo to single point precision float. This way we know what behavior
    to expect from pytigre to Cuda and can change single parameters accordingly.
    """
    geox.cast_to_single()
    # geox.checknans()

    if projections.shape != (geox.angles.shape[0], geox.nDetector[0], geox.nDetector[1]):
        raise ValueError(
            "Expected data shape: "
            + str((geox.angles.shape[0], geox.nDetector[0], geox.nDetector[1]))
            + " not compatible with: "
            + str(projections.shape)
        )

    if kwargs.get("gpuids", None) is None:
        gpuids = GpuIds()
    else:
        gpuids = kwargs["gpuids"]

    # if we have more GPUs than slices to compute, reduce the amount of GPUs. 
    if geo.nVoxel[0]< len(gpuids):
        gpuids.devices = list(gpuids.devices[0:geo.nVoxel[0]])

    # Use backend system for SYCL-only mode
    if USE_BACKEND:
        from .Atb_backend import Atb as Atb_backend
        return Atb_backend(projections, geox, angles, backprojection_type, **kwargs)
    else:
        return _Atb_ext(projections, geox, geox.angles, backprojection_type, geox.mode, gpuids=gpuids)

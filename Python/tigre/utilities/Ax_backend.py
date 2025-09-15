import copy
import numpy as np

# Import both original CUDA and new backend extensions
try:
    from _Ax import _Ax_ext
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA extensions not available")

try:
    from _Ax_backend import Ax_backend as _Ax_backend_ext
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    print("Backend extensions not available")

from .gpu import GpuIds


def Ax(img, geo, angles, projection_type="Siddon", backend=None, device_index=0, **kwargs):
    """
    Forward projection operator Ax for TIGRE with multi-backend support.
    
    Parameters:
    -----------
    img : numpy.ndarray
        3D volume data to project (nVoxel[0] x nVoxel[1] x nVoxel[2])
    geo : tigre.utilities.geometry.Geometry
        TIGRE geometry object defining scan parameters
    angles : numpy.ndarray
        2D array of projection angles (nAngles x 1 or nAngles x 3)
    projection_type : str, optional
        Type of projection algorithm:
        - "Siddon" (default): Siddon ray-tracing
        - "interpolated": Interpolated projections
        - "ray-voxel": Alias for Siddon
    backend : str, optional
        Compute backend to use:
        - None (default): Auto-detect, prefer CUDA if available
        - "CUDA": Use NVIDIA CUDA
        - "SYCL": Use Intel SYCL/oneAPI
        - "CPU": Use CPU implementation
    device_index : int, optional
        Device index to use (default: 0)
    **kwargs : dict
        Additional arguments passed to original CUDA implementation:
        - gpuids : GpuIds object for CUDA backend
        - mode : str, projection mode ("cone" or "parallel")
        
    Returns:
    --------
    numpy.ndarray
        3D projection data (nAngles x nDetector[0] x nDetector[1])
    """
    
    if img.dtype != np.float32:
        raise TypeError("Input data should be float32, not " + str(img.dtype))
    if not np.isreal(img).all():
        raise ValueError("Complex types not compatible for projection.")
    
    geox = copy.deepcopy(geo)
    geox.check_geo(angles)
    geox.cast_to_single()
    
    if img.shape != (geox.nVoxel[0], geox.nVoxel[1], geox.nVoxel[2]):
        raise ValueError(
            "Expected data shape: "
            + str((geox.nVoxel[0], geox.nVoxel[1], geox.nVoxel[2]))
            + " not compatible with: "
            + str(img.shape)
        )
    
    # Backend selection logic
    if backend is None:
        # Auto-detect: prefer CUDA if available, fallback to SYCL
        if CUDA_AVAILABLE:
            backend = "CUDA"
        elif BACKEND_AVAILABLE:
            backend = "SYCL"
        else:
            raise RuntimeError("No compute backend available")
    
    backend = backend.upper()
    
    # Route to appropriate backend
    if backend == "CUDA":
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend requested but not available")
        
        # Use original CUDA implementation
        if kwargs.get("gpuids", None) is None:
            gpuids = GpuIds()
        else:
            gpuids = kwargs["gpuids"]
            
        mode = kwargs.get("mode", "cone")
        return _Ax_ext(img, geox, geox.angles, projection_type, mode, gpuids=gpuids)
    
    elif backend in ["SYCL", "CPU"]:
        if not BACKEND_AVAILABLE:
            raise RuntimeError(f"{backend} backend requested but backend abstraction not available")
        
        # Use new backend abstraction - pass original angles, not geox.angles
        return _Ax_backend_ext(img, geox, angles, projection_type,
                              device_index=device_index, backend=backend)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: CUDA, SYCL, CPU")


def get_available_backends():
    """
    Get list of available compute backends.
    
    Returns:
    --------
    list
        List of available backend strings
    """
    backends = []
    if CUDA_AVAILABLE:
        backends.append("CUDA")
    if BACKEND_AVAILABLE:
        backends.extend(["SYCL", "CPU"])
    return backends
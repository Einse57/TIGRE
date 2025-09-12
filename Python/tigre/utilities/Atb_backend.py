import copy
import numpy as np

# Import both original CUDA and new backend extensions
try:
    from _Atb import _Atb_ext
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA extensions not available")

try:
    from _Atb_backend import _Atb_backend_ext
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    print("Backend extensions not available")

from .gpu import GpuIds


def Atb(projections, geo, angles, backprojection_type="FDK", backend=None, device_index=0, **kwargs):
    """
    Backprojection operator Atb for TIGRE with multi-backend support.
    
    Parameters:
    -----------
    projections : numpy.ndarray
        3D projection data (nAngles x nDetector[0] x nDetector[1])
    geo : tigre.utilities.geometry.Geometry
        TIGRE geometry object defining scan parameters
    angles : numpy.ndarray
        2D array of projection angles (nAngles x 1 or nAngles x 3)
    backprojection_type : str, optional
        Type of backprojection algorithm:
        - "FDK" (default): Feldkamp-Davis-Kress reconstruction
        - "matched": Matched backprojection for iterative methods
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
        3D volume data (nVoxel[0] x nVoxel[1] x nVoxel[2])
    """
    
    if projections.dtype != np.float32:
        raise TypeError("Input data should be float32, not " + str(projections.dtype))
    if not np.isreal(projections).all():
        raise ValueError("Complex types not compatible for back projection.")
    
    geox = copy.deepcopy(geo)
    geox.check_geo(angles)
    geox.cast_to_single()
    
    if projections.shape != (geox.angles.shape[0], geox.nDetector[0], geox.nDetector[1]):
        raise ValueError(
            "Expected data shape: "
            + str((geox.angles.shape[0], geox.nDetector[0], geox.nDetector[1]))
            + " not compatible with: "
            + str(projections.shape)
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
            
        # Reduce GPU count if more GPUs than slices
        if geox.nVoxel[0] < len(gpuids):
            gpuids.devices = list(gpuids.devices[0:geox.nVoxel[0]])
            
        mode = kwargs.get("mode", "cone")
        return _Atb_ext(projections, geox, geox.angles, backprojection_type, mode, gpuids=gpuids)
    
    elif backend in ["SYCL", "CPU"]:
        if not BACKEND_AVAILABLE:
            raise RuntimeError(f"{backend} backend requested but backend abstraction not available")
        
        # Use new backend abstraction
        mode = kwargs.get("mode", "cone")  # For compatibility, not used by backend
        return _Atb_backend_ext(projections, geox, geox.angles, backprojection_type, mode,
                               backend=backend, device_index=device_index)
    
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
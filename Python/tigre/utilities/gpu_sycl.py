"""
SYCL-only stub for gpu.py functionality

This provides minimal GPU device management for SYCL builds
without requiring the full CUDA-based _gpuUtils extension.
"""

class GpuIds(object):
    """
    Simplified GpuIds class for SYCL-only builds
    
    In SYCL mode, device selection is handled by the SYCL runtime
    and the device_index parameter in the backend functions.
    """
    
    def __init__(self, name=None):
        """
        Initialize with device name (ignored in SYCL builds)
        """
        self.name = name if name else "SYCL Device"
        self.deviceids = [0]  # Default to first SYCL device
        
    def __len__(self):
        return len(self.deviceids)
        
    def __getitem__(self, index):
        return self.deviceids[index]
        
    def __iter__(self):
        return iter(self.deviceids)


def getGpuNames():
    """
    Get available GPU names (SYCL version)
    
    In full SYCL implementation, this would query SYCL devices.
    For now, returns a placeholder.
    """
    return ["Intel Graphics (SYCL)"]


def getGpuIds(name=""):
    """
    Get GPU IDs by name (SYCL version)
    
    Returns GpuIds object for SYCL device selection.
    """
    return GpuIds(name)


def numGpuIds():
    """
    Get number of available GPUs (SYCL version)
    """
    return 1  # Simplified for SYCL-only builds
"""
1. Check the names of installed GPUs.
   names = getGpuNames()
   print(names)
2. Select the name of the GPUs that you want your program run on.
   gpuids = getGpuIds('GeForce RTX 2080 Ti')
3. Pass the gpuids instance to Tigre functions.
"""

# Try to import CUDA-based gpuUtils, fall back to SYCL version
try:
    import _gpuUtils as gpuUtils
    SYCL_ONLY_MODE = False
except ImportError:
    # SYCL-only mode - no CUDA available
    SYCL_ONLY_MODE = True
    print("⚠️  CUDA _gpuUtils not available - using SYCL device management")

import copy

class GpuIds(object):
    """
    A class that holds the IDs and their (common) name
    ```
    gpuids = GpuIds('GeForce RTX 2080 Ti')
    ```
    If name is ""/None, IDs of all GPUs are hold.
    ```
    gpuids = GpuIds()
    ```
    """

    def __init__(self, nameGPU=None):  # noqa: N803
        if nameGPU is None:
            self.name = ""
        else:
            self.name = nameGPU
            
        if SYCL_ONLY_MODE:
            # SYCL-only mode: simple device management
            self.devices = [0]  # Default to first SYCL device
        else:
            # CUDA mode: use gpuUtils
            self.devices = gpuUtils.getGpuIdList(self.name)

    def __len__(self):
        return len(self.devices)

    def __str__(self):
        dictTemp = {
            "name": self.name,
            "devices": self.devices,
        }
        return dictTemp.__str__()
    def __getitem__(self, key):
        single_gpu = copy.deepcopy(self)
        single_gpu.devices = [self.devices[key]]
        return single_gpu

def getGpuNames():
    """Returns a list of all installed GPUs."""
    if SYCL_ONLY_MODE:
        # SYCL-only mode: return SYCL device info
        return ["Intel Graphics (SYCL)"]
    else:
        # CUDA mode
        return gpuUtils.getGpuNames()


# def getGpuCount():
#     """Returns the number of installed GPUs."""
#     if SYCL_ONLY_MODE:
#         return 1
#     else:
#         return gpuUtils.getGpuCount()


def getGpuIds(gpuName=None):  # noqa: N803
    """Returns the GpuIds object, which contains the IDs of devices whose name matches gpuName."""
    gpuids = GpuIds(gpuName)
    return gpuids

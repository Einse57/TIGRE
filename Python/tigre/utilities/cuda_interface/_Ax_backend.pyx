# distutils: language=c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libc.stdlib cimport malloc, free
import ctypes

# Import TIGRE types
cdef extern from "tigre/backend/device_backend.hpp" namespace "tigre":
    # Geometry structure (matches actual C++ structure)
    cdef struct Geometry:
        # Voxel parameters
        int nVoxelX, nVoxelY, nVoxelZ
        float sVoxelX, sVoxelY, sVoxelZ
        float dVoxelX, dVoxelY, dVoxelZ
        float *offOrigX, *offOrigY, *offOrigZ
        
        # Detector parameters
        int nDetecU, nDetecV
        float sDetecU, sDetecV
        float dDetecU, dDetecV
        float *offDetecU, *offDetecV
        
        # Source-detector geometry
        float *DSO   # Distance Source-Object
        float *DSD   # Distance Source-Detector
        
        # Rotation parameters
        float *dRoll, *dPitch, *dYaw
        float *COR   # Center of Rotation correction
        
        # Other parameters
        float unitX, unitY, unitZ
        float alpha, theta, psi  # Euler angles
        float maxLength
        float accuracy

cdef extern from "tigre/backend/device_backend.hpp" namespace "tigre":
    # View parameters structure
    cdef struct Views:
        vector[float] theta     # Projection angles
        vector[float] off_u     # U-direction offsets
        vector[float] off_v     # V-direction offsets
    
    # Device buffer abstraction
    cdef struct Buffer:
        void* ptr
        size_t bytes
    
    # Backend types
    cdef enum class BackendKind:
        CUDA, SYCL, CPU
    
    # Abstract backend interface
    cdef cppclass DeviceBackend:
        string name()
        void synchronize()
        Buffer malloc_bytes(size_t size_bytes)
        void free(Buffer& buffer)
        void memcpy_h2d(Buffer& dst_buffer, const void* src_host_ptr, size_t size_bytes)
        void memcpy_d2h(void* dst_host_ptr, const Buffer& src_buffer, size_t size_bytes)
        void memset_zero(Buffer& buffer)
        void launch_forward_projector(const Geometry& geo, const Views& views, 
                                     const Buffer& volume_buffer, Buffer& projection_buffer,
                                     int num_angles, int projection_type)
    
    # Factory and utility functions
    unique_ptr[DeviceBackend] make_backend(BackendKind kind, int device_index)
    string backend_kind_to_string(BackendKind kind)
    BackendKind backend_kind_from_string(const string& str)

def Ax_backend(np.ndarray[np.float32_t, ndim=3] img not None,
               geometry, np.ndarray[np.float32_t, ndim=1] angles not None,
               mode="interpolated", 
               int device_index=0,
               str backend="auto"):
    """
    Forward projection operator using device backends (SYCL/CUDA/CPU)
    """
    
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # All cdef declarations must be at the start
    cdef int total_projections = len(angles)
    cdef Geometry c_geometry
    cdef string backend_str
    cdef BackendKind backend_kind  
    cdef unique_ptr[DeviceBackend] device_backend
    cdef float* c_img
    cdef size_t volume_size
    cdef int nDetectorPixels
    cdef size_t projection_size
    cdef Buffer volume_buffer
    cdef Buffer projection_buffer
    cdef float* c_angles
    cdef Views views
    cdef int i
    cdef int projection_type
    cdef float* c_projections_flat
    cdef np.npy_intp shape[3]
    
    # Geometry pointer fields (need to be allocated)
    cdef float* dso_ptr
    cdef float* dsd_ptr
    cdef float* offOrigX_ptr
    cdef float* offOrigY_ptr
    cdef float* offOrigZ_ptr
    cdef float* offDetecU_ptr
    cdef float* offDetecV_ptr
    cdef float* cor_ptr
    
    if total_projections == 0:
        raise ValueError("No projection angles provided")
    
    # Allocate memory for pointer fields (single values per field)
    dso_ptr = <float*> malloc(sizeof(float))
    dsd_ptr = <float*> malloc(sizeof(float))
    offOrigX_ptr = <float*> malloc(sizeof(float))
    offOrigY_ptr = <float*> malloc(sizeof(float))
    offOrigZ_ptr = <float*> malloc(sizeof(float))
    offDetecU_ptr = <float*> malloc(sizeof(float))
    offDetecV_ptr = <float*> malloc(sizeof(float))
    cor_ptr = <float*> malloc(sizeof(float))
    
    if not (dso_ptr and dsd_ptr and offOrigX_ptr and offOrigY_ptr and offOrigZ_ptr and offDetecU_ptr and offDetecV_ptr and cor_ptr):
        raise MemoryError("Failed to allocate memory for geometry parameters")
    
    # Copy geometry parameters to structure with correct field names
    # Voxel parameters
    c_geometry.nVoxelX = geometry.nVoxel[0]
    c_geometry.nVoxelY = geometry.nVoxel[1] 
    c_geometry.nVoxelZ = geometry.nVoxel[2]
    c_geometry.sVoxelX = geometry.sVoxel[0]
    c_geometry.sVoxelY = geometry.sVoxel[1]
    c_geometry.sVoxelZ = geometry.sVoxel[2]
    c_geometry.dVoxelX = geometry.dVoxel[0]
    c_geometry.dVoxelY = geometry.dVoxel[1]
    c_geometry.dVoxelZ = geometry.dVoxel[2]
    
    # Detector parameters
    c_geometry.nDetecU = geometry.nDetector[0]
    c_geometry.nDetecV = geometry.nDetector[1]
    c_geometry.sDetecU = geometry.sDetector[0]
    c_geometry.sDetecV = geometry.sDetector[1]
    c_geometry.dDetecU = geometry.dDetector[0]
    c_geometry.dDetecV = geometry.dDetector[1]
    
    # Set pointer field values
    dso_ptr[0] = geometry.DSO
    dsd_ptr[0] = geometry.DSD
    offOrigX_ptr[0] = geometry.offOrigin[0] if hasattr(geometry, 'offOrigin') else 0.0
    offOrigY_ptr[0] = geometry.offOrigin[1] if hasattr(geometry, 'offOrigin') else 0.0
    offOrigZ_ptr[0] = geometry.offOrigin[2] if hasattr(geometry, 'offOrigin') else 0.0
    offDetecU_ptr[0] = geometry.offDetector[0] if hasattr(geometry, 'offDetector') else 0.0
    offDetecV_ptr[0] = geometry.offDetector[1] if hasattr(geometry, 'offDetector') else 0.0
    cor_ptr[0] = geometry.COR if hasattr(geometry, 'COR') else 0.0
    
    # Assign pointers to geometry structure
    c_geometry.DSO = dso_ptr
    c_geometry.DSD = dsd_ptr
    c_geometry.offOrigX = offOrigX_ptr
    c_geometry.offOrigY = offOrigY_ptr
    c_geometry.offOrigZ = offOrigZ_ptr
    c_geometry.offDetecU = offDetecU_ptr
    c_geometry.offDetecV = offDetecV_ptr
    c_geometry.COR = cor_ptr
    
    # Set other parameters
    c_geometry.accuracy = getattr(geometry, 'accuracy', 0.5)
    
    # Convert backend string to enum
    backend_str = backend.encode('utf-8')
    backend_kind = backend_kind_from_string(backend_str)
    
    # Create backend instance
    device_backend = make_backend(backend_kind, device_index)
    if not device_backend:
        raise RuntimeError(f"Failed to create {backend} backend")
    
    print(f"Using backend: {device_backend.get().name().decode('utf-8')}")
    
    # Ensure input image is contiguous
    img = np.ascontiguousarray(img)
    c_img = <float*> img.data
    
    # Calculate buffer sizes
    volume_size = img.size * sizeof(float)
    nDetectorPixels = geometry.nDetector[0] * geometry.nDetector[1] 
    projection_size = total_projections * nDetectorPixels * sizeof(float)
    
    # Allocate device buffers
    volume_buffer = device_backend.get().malloc_bytes(volume_size)
    projection_buffer = device_backend.get().malloc_bytes(projection_size)
    
    # Prepare angles for Views structure
    angles = np.ascontiguousarray(angles)
    c_angles = <float*> angles.data
    
    # Create Views structure  
    for i in range(total_projections):
        views.theta.push_back(c_angles[i])
        views.off_u.push_back(0.0)  # Default offset
        views.off_v.push_back(0.0)  # Default offset
    
    # Set projection type (0=siddon, 1=interpolated)
    projection_type = 1 if mode == "interpolated" else 0
    
    # Allocate host output memory
    c_projections_flat = <float*> malloc(projection_size)
    if not c_projections_flat:
        raise MemoryError("Error allocating output memory")
    
    try:
        # Copy volume data to device
        device_backend.get().memcpy_h2d(volume_buffer, c_img, volume_size)
        
        # Zero out projection buffer
        device_backend.get().memset_zero(projection_buffer)
        
        # Perform forward projection using backend
        device_backend.get().launch_forward_projector(c_geometry, views, volume_buffer, 
                                                     projection_buffer, total_projections, 
                                                     projection_type)
        
        # Synchronize device
        device_backend.get().synchronize()
        
        # Copy projection data back from device
        device_backend.get().memcpy_d2h(c_projections_flat, projection_buffer, projection_size)
        
        # Create output numpy array
        shape[0] = <np.npy_intp> total_projections
        shape[1] = <np.npy_intp> (<np.npy_long>(geometry.nDetector[0]))
        shape[2] = <np.npy_intp> (<np.npy_long>(geometry.nDetector[1]))
        
        projections = np.PyArray_SimpleNewFromData(3, shape, np.NPY_FLOAT32, c_projections_flat)
        np.PyArray_ENABLEFLAGS(projections, np.NPY_ARRAY_OWNDATA)  # Transfer ownership
        
    except Exception as e:
        free(c_projections_flat)
        raise RuntimeError(f"Forward projection failed: {str(e)}")
    
    finally:
        # Clean up device buffers
        device_backend.get().free(volume_buffer)
        device_backend.get().free(projection_buffer)
        
        # Free geometry pointer fields
        free(dso_ptr)
        free(dsd_ptr)
        free(offOrigX_ptr)
        free(offOrigY_ptr)
        free(offOrigZ_ptr)
        free(offDetecU_ptr)
        free(offDetecV_ptr)
        free(cor_ptr)
    
    return projections
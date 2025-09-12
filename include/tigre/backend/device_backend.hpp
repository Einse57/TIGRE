#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tigre {

/**
 * @brief Geometry structure that matches TIGRE's geometry representation
 * Simplified version compatible with both CUDA and SYCL implementations
 */
struct Geometry {
    // Voxel parameters
    int nVoxelX, nVoxelY, nVoxelZ;
    float sVoxelX, sVoxelY, sVoxelZ;
    float dVoxelX, dVoxelY, dVoxelZ;
    float *offOrigX, *offOrigY, *offOrigZ;
    
    // Detector parameters
    int nDetecU, nDetecV;
    float sDetecU, sDetecV;
    float dDetecU, dDetecV;
    float *offDetecU, *offDetecV;
    
    // Source-detector geometry
    float *DSO;  // Distance Source-Object
    float *DSD;  // Distance Source-Detector
    
    // Rotation parameters
    float *dRoll, *dPitch, *dYaw;
    float *COR;  // Center of Rotation correction
    
    // Other parameters
    float unitX, unitY, unitZ;
    float alpha, theta, psi;  // Euler angles
    float maxLength;
    float accuracy;
};

/**
 * @brief View parameters for projection angles
 */
struct Views {
    std::vector<float> theta;     // Projection angles
    std::vector<float> off_u;     // U-direction offsets  
    std::vector<float> off_v;     // V-direction offsets
};

/**
 * @brief Device memory buffer abstraction
 */
struct Buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    
    Buffer() = default;
    Buffer(void* p, size_t s) : ptr(p), bytes(s) {}
    
    // Non-copyable but movable
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
        other.ptr = nullptr;
        other.bytes = 0;
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
};

/**
 * @brief Backend types supported by TIGRE
 */
enum class BackendKind {
    CUDA,   // NVIDIA CUDA backend
    SYCL,   // Intel oneAPI SYCL backend  
    CPU     // CPU-only backend
};

/**
 * @brief Abstract base class for device computation backends
 * 
 * This interface provides a unified API for different compute backends
 * (CUDA, SYCL, CPU) to perform tomographic reconstruction operations.
 */
class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;
    
    /**
     * @brief Get backend name/description
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Synchronize with device (wait for all operations to complete)
     */
    virtual void synchronize() = 0;
    
    /**
     * @brief Allocate device memory buffer
     * @param size_bytes Number of bytes to allocate
     * @return Buffer object containing device pointer
     */
    virtual Buffer malloc_bytes(size_t size_bytes) = 0;
    
    /**
     * @brief Free device memory buffer
     * @param buffer Buffer to free (will be reset to null)
     */
    virtual void free(Buffer& buffer) = 0;
    
    /**
     * @brief Copy data from host to device
     * @param dst_buffer Destination device buffer
     * @param src_host_ptr Source host pointer
     * @param size_bytes Number of bytes to copy
     */
    virtual void memcpy_h2d(Buffer& dst_buffer, const void* src_host_ptr, size_t size_bytes) = 0;
    
    /**
     * @brief Copy data from device to host
     * @param dst_host_ptr Destination host pointer
     * @param src_buffer Source device buffer
     * @param size_bytes Number of bytes to copy
     */
    virtual void memcpy_d2h(void* dst_host_ptr, const Buffer& src_buffer, size_t size_bytes) = 0;
    
    /**
     * @brief Set device memory to zero
     * @param buffer Device buffer to zero out
     */
    virtual void memset_zero(Buffer& buffer) = 0;
    
    /**
     * @brief Launch forward projector kernel (Ax operation)
     * @param geo Geometry parameters
     * @param views View/angle parameters
     * @param volume_buffer Input volume data on device
     * @param projection_buffer Output projection data on device
     * @param num_angles Number of projection angles
     * @param projection_type Type of projector (0=Siddon, 1=interpolated, etc.)
     */
    virtual void launch_forward_projector(
        const Geometry& geo,
        const Views& views, 
        const Buffer& volume_buffer,
        Buffer& projection_buffer,
        int num_angles,
        int projection_type
    ) = 0;
    
    /**
     * @brief Launch back projector kernel (Atb operation)
     * @param geo Geometry parameters
     * @param views View/angle parameters  
     * @param projection_buffer Input projection data on device
     * @param volume_buffer Output volume data on device
     * @param num_angles Number of projection angles
     * @param backprojection_type Type of backprojector (0=FDK, 1=matched, etc.)
     */
    virtual void launch_back_projector(
        const Geometry& geo,
        const Views& views,
        const Buffer& projection_buffer, 
        Buffer& volume_buffer,
        int num_angles,
        int backprojection_type
    ) = 0;
};

/**
 * @brief Factory function to create backend instances
 * @param kind Type of backend to create
 * @param device_index Device index (for multi-device systems)
 * @return Unique pointer to backend instance
 */
std::unique_ptr<DeviceBackend> make_backend(BackendKind kind, int device_index = 0);

/**
 * @brief Get string representation of backend kind
 */
std::string backend_kind_to_string(BackendKind kind);

/**
 * @brief Parse backend kind from string
 */
BackendKind backend_kind_from_string(const std::string& str);

} // namespace tigre
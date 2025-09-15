#include "device_backend_sycl.hpp"
#include "siddon_projection_sycl_fixed.hpp"
#include "fdk_backprojection_sycl.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>

#ifdef TIGRE_BUILD_SYCL_BACKEND
#include <sycl/sycl.hpp>
#endif

namespace tigre {

DeviceBackendSYCL::DeviceBackendSYCL(int device_index) 
    : device_index_(device_index) {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    init_device(device_index);
#else
    throw std::runtime_error("TIGRE was not compiled with SYCL support. "
                           "Please rebuild with TIGRE_BUILD_SYCL_BACKEND=ON");
#endif
}

DeviceBackendSYCL::~DeviceBackendSYCL() = default;

std::string DeviceBackendSYCL::name() const {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    std::stringstream ss;
    ss << "SYCL(oneAPI/Level Zero) on " << device_.get_info<sycl::info::device::name>();
    return ss.str();
#else
    return "SYCL(disabled)";
#endif
}

void DeviceBackendSYCL::synchronize() {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    queue_.wait_and_throw();
#endif
}

Buffer DeviceBackendSYCL::malloc_bytes(size_t size_bytes) {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    try {
        // Use SYCL Unified Shared Memory (USM) for device allocation
        void* ptr = sycl::malloc_device(size_bytes, queue_);
        if (!ptr) {
            throw std::runtime_error("Failed to allocate " + std::to_string(size_bytes) + " bytes on SYCL device");
        }
        return Buffer(ptr, size_bytes);
    } catch (const sycl::exception& e) {
        throw std::runtime_error("SYCL malloc failed: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}

void DeviceBackendSYCL::free(Buffer& buffer) {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    if (buffer.ptr) {
        sycl::free(buffer.ptr, queue_);
        buffer.ptr = nullptr;
        buffer.bytes = 0;
    }
#endif
}

void DeviceBackendSYCL::memcpy_h2d(Buffer& dst_buffer, const void* src_host_ptr, size_t size_bytes) {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    if (size_bytes > dst_buffer.bytes) {
        throw std::runtime_error("memcpy_h2d: size_bytes exceeds buffer size");
    }
    
    try {
        queue_.memcpy(dst_buffer.ptr, src_host_ptr, size_bytes);
        queue_.wait();
    } catch (const sycl::exception& e) {
        throw std::runtime_error("SYCL memcpy H2D failed: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}

void DeviceBackendSYCL::memcpy_d2h(void* dst_host_ptr, const Buffer& src_buffer, size_t size_bytes) {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    if (size_bytes > src_buffer.bytes) {
        throw std::runtime_error("memcpy_d2h: size_bytes exceeds buffer size");
    }
    
    try {
        queue_.memcpy(dst_host_ptr, src_buffer.ptr, size_bytes);
        queue_.wait();
    } catch (const sycl::exception& e) {
        throw std::runtime_error("SYCL memcpy D2H failed: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}

void DeviceBackendSYCL::memset_zero(Buffer& buffer) {
#ifdef TIGRE_BUILD_SYCL_BACKEND
    try {
        queue_.memset(buffer.ptr, 0, buffer.bytes);
        queue_.wait();
    } catch (const sycl::exception& e) {
        throw std::runtime_error("SYCL memset failed: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}

void DeviceBackendSYCL::launch_forward_projector(
    const Geometry& geo,
    const Views& views, 
    const Buffer& volume_buffer,
    Buffer& projection_buffer,
    int num_angles,
    int projection_type) {
    
#ifdef TIGRE_BUILD_SYCL_BACKEND
    // Dispatch to appropriate projection kernel based on type
    switch (projection_type) {
        case 0:  // Siddon ray-voxel intersection
            launch_siddon_projection_sycl(geo, views, volume_buffer, projection_buffer, num_angles);
            break;
        case 1:  // Interpolated projection (texture-based equivalent)
            // TODO: Implement interpolated projection
            throw std::runtime_error("Interpolated projection not yet implemented in SYCL backend");
            break;
        default:
            throw std::runtime_error("Unknown projection type: " + std::to_string(projection_type));
    }
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}

void DeviceBackendSYCL::launch_back_projector(
    const Geometry& geo,
    const Views& views,
    const Buffer& projection_buffer, 
    Buffer& volume_buffer,
    int num_angles,
    int backprojection_type) {
    
#ifdef TIGRE_BUILD_SYCL_BACKEND
    // Dispatch to appropriate backprojection kernel based on type
    switch (backprojection_type) {
        case 0:  // FDK backprojection
            launch_fdk_backprojection_sycl(geo, views, projection_buffer, volume_buffer, num_angles);
            break;
        case 1:  // Matched backprojection
            // TODO: Implement matched backprojection
            throw std::runtime_error("Matched backprojection not yet implemented in SYCL backend");
            break;
        default:
            throw std::runtime_error("Unknown backprojection type: " + std::to_string(backprojection_type));
    }
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}

#ifdef TIGRE_BUILD_SYCL_BACKEND
void DeviceBackendSYCL::init_device(int device_index) {
    try {
        // Get available SYCL devices
        auto devices = sycl::device::get_devices();
        
        if (devices.empty()) {
            throw std::runtime_error("No SYCL devices found");
        }
        
        // Try to find a GPU device first, fallback to any available device
        sycl::device selected_device;
        bool found_gpu = false;
        
        for (const auto& dev : devices) {
            if (dev.is_gpu()) {
                selected_device = dev;
                found_gpu = true;
                break;
            }
        }
        
        if (!found_gpu) {
            std::cout << "Warning: No GPU devices found, using first available device" << std::endl;
            selected_device = devices[0];
        }
        
        // Create context and queue
        device_ = selected_device;
        context_ = sycl::context(device_);
        queue_ = sycl::queue(context_, device_);
        
        std::cout << "Initialized SYCL backend on device: " 
                  << device_.get_info<sycl::info::device::name>() << std::endl;
                  
    } catch (const sycl::exception& e) {
        throw std::runtime_error("Failed to initialize SYCL device: " + std::string(e.what()));
    }
}

void DeviceBackendSYCL::launch_siddon_projection_sycl(
    const Geometry& geo,
    const Views& views,
    const Buffer& volume_buffer,
    Buffer& projection_buffer,
    int num_angles) {
    
#ifdef TIGRE_BUILD_SYCL_BACKEND
    // Cast buffer pointers to appropriate types
    const float* volume_data = static_cast<const float*>(volume_buffer.ptr);
    float* projection_data = static_cast<float*>(projection_buffer.ptr);
    
    // Launch the SYCL Siddon projection kernel
    sycl_kernels::SiddonProjectionKernel::launch(
        queue_, geo, views, volume_data, projection_data, num_angles
    );
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}

void DeviceBackendSYCL::launch_fdk_backprojection_sycl(
    const Geometry& geo,
    const Views& views,
    const Buffer& projection_buffer,
    Buffer& volume_buffer,
    int num_angles) {
    
#ifdef TIGRE_BUILD_SYCL_BACKEND
    // Cast buffer pointers to appropriate types
    const float* projection_data = static_cast<const float*>(projection_buffer.ptr);
    float* volume_data = static_cast<float*>(volume_buffer.ptr);
    
    // Launch the SYCL FDK backprojection kernel
    sycl_kernels::FDKBackprojectionKernel::launch(
        queue_, geo, views, projection_data, volume_data, num_angles
    );
#else
    throw std::runtime_error("SYCL backend not available");
#endif
}
#endif

// Factory function for external linkage (no SYCL headers exposed)
std::unique_ptr<tigre::DeviceBackend> create_sycl_backend(int device_index) {
    return std::make_unique<tigre::DeviceBackendSYCL>(device_index);
}

} // namespace tigre
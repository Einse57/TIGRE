#pragma once

#include "tigre/backend/device_backend.hpp"

// Only include SYCL headers when building with SYCL support
#ifdef TIGRE_BUILD_SYCL_BACKEND
#include <sycl/sycl.hpp>
#else
// Forward declarations when SYCL is not available
namespace sycl {
    class queue;
    class device;
    class context;
}
#endif

namespace tigre {

/**
 * @brief SYCL backend implementation using Intel oneAPI DPC++
 * 
 * This backend uses Intel's SYCL implementation to run on Intel GPUs,
 * including integrated graphics (iGPU) and discrete Arc GPUs.
 */
class DeviceBackendSYCL final : public DeviceBackend {
public:
    /**
     * @brief Construct SYCL backend
     * @param device_index Index of SYCL device to use (0=default)
     */
    explicit DeviceBackendSYCL(int device_index = 0);
    
    /**
     * @brief Destructor - ensures proper cleanup
     */
    ~DeviceBackendSYCL() override;
    
    // DeviceBackend interface implementation
    std::string name() const override;
    void synchronize() override;
    Buffer malloc_bytes(size_t size_bytes) override;
    void free(Buffer& buffer) override;
    void memcpy_h2d(Buffer& dst_buffer, const void* src_host_ptr, size_t size_bytes) override;
    void memcpy_d2h(void* dst_host_ptr, const Buffer& src_buffer, size_t size_bytes) override;
    void memset_zero(Buffer& buffer) override;
    
    void launch_forward_projector(
        const Geometry& geo,
        const Views& views, 
        const Buffer& volume_buffer,
        Buffer& projection_buffer,
        int num_angles,
        int projection_type
    ) override;
    
    void launch_back_projector(
        const Geometry& geo,
        const Views& views,
        const Buffer& projection_buffer, 
        Buffer& volume_buffer,
        int num_angles,
        int backprojection_type
    ) override;

private:
#ifdef TIGRE_BUILD_SYCL_BACKEND
    sycl::queue queue_;           // SYCL command queue
    sycl::device device_;         // SYCL device
    sycl::context context_;       // SYCL context
#endif
    int device_index_;            // Device index for identification
    
    // Private helper methods
    void init_device(int device_index);
    void launch_siddon_projection_sycl(
        const Geometry& geo,
        const Views& views,
        const Buffer& volume_buffer,
        Buffer& projection_buffer,
        int num_angles
    );
    void launch_fdk_backprojection_sycl(
        const Geometry& geo,
        const Views& views,
        const Buffer& projection_buffer,
        Buffer& volume_buffer,
        int num_angles
    );
};

} // namespace tigre
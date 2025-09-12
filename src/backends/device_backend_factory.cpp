#include "tigre/backend/device_backend.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

#if TIGRE_BUILD_SYCL_BACKEND
#include "../src/backends/sycl/device_backend_sycl.hpp"
#endif

namespace tigre {

std::unique_ptr<DeviceBackend> make_backend(BackendKind kind, int device_index) {
    switch (kind) {
#if TIGRE_BUILD_SYCL_BACKEND
        case BackendKind::SYCL:
            return std::make_unique<DeviceBackendSYCL>(device_index);
#endif
            
        case BackendKind::CUDA:
            // TODO: Implement CUDA backend wrapper
            throw std::runtime_error("CUDA backend not yet implemented in new abstraction layer");
            
        case BackendKind::CPU:
            // TODO: Implement CPU backend
            throw std::runtime_error("CPU backend not yet implemented");
            
        default:
            throw std::runtime_error("Unknown backend kind or backend not compiled");
    }
}

std::string backend_kind_to_string(BackendKind kind) {
    switch (kind) {
        case BackendKind::CUDA:
            return "cuda";
        case BackendKind::SYCL:
            return "sycl";
        case BackendKind::CPU:
            return "cpu";
        default:
            return "unknown";
    }
}

BackendKind backend_kind_from_string(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    
    if (lower_str == "cuda") {
        return BackendKind::CUDA;
    } else if (lower_str == "sycl") {
        return BackendKind::SYCL;
    } else if (lower_str == "cpu") {
        return BackendKind::CPU;
    } else {
        throw std::runtime_error("Unknown backend kind: " + str);
    }
}

} // namespace tigre
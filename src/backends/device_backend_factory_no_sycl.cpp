#include "tigre/backend/device_backend.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace tigre {

// Forward declarations to avoid including SYCL headers in factory
#if TIGRE_BUILD_SYCL_BACKEND
// External function to create SYCL backend - implemented in pre-compiled object
extern std::unique_ptr<DeviceBackend> create_sycl_backend(int device_index);
#endif

std::unique_ptr<DeviceBackend> make_backend(BackendKind kind, int device_index) {
    switch (kind) {
#if TIGRE_BUILD_SYCL_BACKEND
        case BackendKind::SYCL:
            return create_sycl_backend(device_index);
#endif
            
        case BackendKind::CUDA:
            // TODO: Implement CUDA backend wrapper
            throw std::runtime_error("CUDA backend not yet implemented in new abstraction layer");
            
        case BackendKind::CPU:
            // TODO: Implement CPU backend
            throw std::runtime_error("CPU backend not yet implemented");
            
        default:
            throw std::runtime_error("Unknown backend kind");
    }
}

BackendKind backend_kind_from_string(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    if (lower_str == "sycl") {
        return BackendKind::SYCL;
    } else if (lower_str == "cuda") {
        return BackendKind::CUDA;
    } else if (lower_str == "cpu") {
        return BackendKind::CPU;
    } else {
        throw std::runtime_error("Unknown backend kind: " + str);
    }
}

std::string backend_kind_to_string(BackendKind kind) {
    switch (kind) {
        case BackendKind::SYCL:  return "SYCL";
        case BackendKind::CUDA:  return "CUDA";  
        case BackendKind::CPU:   return "CPU";
        default: return "UNKNOWN";
    }
}

std::vector<BackendKind> get_available_backends() {
    std::vector<BackendKind> backends;
    
#if TIGRE_BUILD_SYCL_BACKEND
    backends.push_back(BackendKind::SYCL);
#endif
    
    // TODO: Add CUDA detection
    // backends.push_back(BackendKind::CUDA);
    
    // TODO: Add CPU backend
    // backends.push_back(BackendKind::CPU);
    
    return backends;
}

} // namespace tigre
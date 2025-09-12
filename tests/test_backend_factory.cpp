#include "tigre/backend/device_backend.hpp"
#include <iostream>
#include <string>

int main() {
    std::cout << "Testing TIGRE backend factory..." << std::endl;
    
    try {
        // Test backend kind string functions
        auto cuda_str = tigre::backend_kind_to_string(tigre::BackendKind::CUDA);
        auto sycl_str = tigre::backend_kind_to_string(tigre::BackendKind::SYCL);
        auto cpu_str = tigre::backend_kind_to_string(tigre::BackendKind::CPU);
        
        std::cout << "Backend kinds: " << cuda_str << ", " << sycl_str << ", " << cpu_str << std::endl;
        
        // Test string to backend kind parsing
        auto cuda_kind = tigre::backend_kind_from_string("cuda");
        auto sycl_kind = tigre::backend_kind_from_string("sycl");
        auto cpu_kind = tigre::backend_kind_from_string("cpu");
        
        bool conversion_correct = (cuda_kind == tigre::BackendKind::CUDA) &&
                                (sycl_kind == tigre::BackendKind::SYCL) &&
                                (cpu_kind == tigre::BackendKind::CPU);
        
        if (conversion_correct) {
            std::cout << "String conversion tests PASSED" << std::endl;
        } else {
            std::cout << "String conversion tests FAILED" << std::endl;
            return 1;
        }
        
        // Test backend creation (this will fail gracefully if SYCL not available)
        try {
            auto sycl_backend = tigre::make_backend(tigre::BackendKind::SYCL, 0);
            std::cout << "SYCL backend created: " << sycl_backend->name() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "SYCL backend creation failed (expected if not built with SYCL): " << e.what() << std::endl;
        }
        
        std::cout << "Backend factory test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
#include "tigre/backend/device_backend.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>

int main() {
    try {
        std::cout << "Testing TIGRE SYCL Backend..." << std::endl;
        
        // Create SYCL backend
        auto backend = tigre::make_backend(tigre::BackendKind::SYCL, 0);
        std::cout << "Backend: " << backend->name() << std::endl;
        
        // Test basic memory operations
        const size_t test_size = 1024 * sizeof(float);
        std::cout << "Testing memory allocation (" << test_size << " bytes)..." << std::endl;
        
        auto buffer = backend->malloc_bytes(test_size);
        std::cout << "Memory allocated successfully" << std::endl;
        
        // Create test data
        std::vector<float> host_data(1024);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (size_t i = 0; i < host_data.size(); ++i) {
            host_data[i] = dis(gen);
        }
        
        // Test host-to-device copy
        std::cout << "Testing host-to-device copy..." << std::endl;
        backend->memcpy_h2d(buffer, host_data.data(), test_size);
        std::cout << "H2D copy successful" << std::endl;
        
        // Test device-to-host copy
        std::cout << "Testing device-to-host copy..." << std::endl;
        std::vector<float> result_data(1024);
        backend->memcpy_d2h(result_data.data(), buffer, test_size);
        std::cout << "D2H copy successful" << std::endl;
        
        // Verify data integrity
        bool data_correct = true;
        for (size_t i = 0; i < host_data.size(); ++i) {
            if (std::abs(host_data[i] - result_data[i]) > 1e-6f) {
                data_correct = false;
                break;
            }
        }
        
        if (data_correct) {
            std::cout << "Data integrity test PASSED" << std::endl;
        } else {
            std::cout << "Data integrity test FAILED" << std::endl;
            return 1;
        }
        
        // Test synchronization
        std::cout << "Testing synchronization..." << std::endl;
        backend->synchronize();
        std::cout << "Synchronization successful" << std::endl;
        
        // Cleanup
        backend->free(buffer);
        std::cout << "Memory freed successfully" << std::endl;
        
        std::cout << "All SYCL backend tests PASSED!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
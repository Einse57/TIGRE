#include <iostream>
#include <memory>

// Mock implementation for testing without SYCL
namespace tigre {

enum class BackendKind {
    SYCL, CUDA, CPU  
};

struct Buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
};

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;
    virtual std::string name() const = 0;
    virtual void synchronize() = 0;
    virtual Buffer malloc_bytes(size_t bytes) = 0;
    virtual void free(Buffer& buffer) = 0;
    virtual void memcpy_h2d(Buffer& dst, const void* src, size_t bytes) = 0;
    virtual void memcpy_d2h(void* dst, const Buffer& src, size_t bytes) = 0;
    virtual void memset_zero(Buffer& buffer) = 0;
};

class MockBackend : public DeviceBackend {
public:
    std::string name() const override { return "Mock SYCL Backend"; }
    void synchronize() override {}
    Buffer malloc_bytes(size_t bytes) override {
        return Buffer{malloc(bytes), bytes};
    }
    void free(Buffer& buffer) override {
        if (buffer.ptr) {
            ::free(buffer.ptr);
            buffer.ptr = nullptr;
            buffer.bytes = 0;
        }
    }
    void memcpy_h2d(Buffer& dst, const void* src, size_t bytes) override {
        memcpy(dst.ptr, src, bytes);
    }
    void memcpy_d2h(void* dst, const Buffer& src, size_t bytes) override {
        memcpy(dst, src.ptr, bytes);
    }
    void memset_zero(Buffer& buffer) override {
        memset(buffer.ptr, 0, buffer.bytes);
    }
};

std::unique_ptr<DeviceBackend> make_backend(BackendKind kind, int device_index = 0) {
    return std::make_unique<MockBackend>();
}

}

int main() {
    std::cout << "Testing TIGRE Mock Backend..." << std::endl;
    
    try {
        auto backend = tigre::make_backend(tigre::BackendKind::SYCL);
        std::cout << "Backend: " << backend->name() << std::endl;
        
        // Test basic memory operations
        const size_t test_size = 1024 * sizeof(float);
        auto buffer = backend->malloc_bytes(test_size);
        std::cout << "Memory allocated: " << buffer.bytes << " bytes" << std::endl;
        
        backend->memset_zero(buffer);
        backend->synchronize();
        backend->free(buffer);
        
        std::cout << "Mock backend test PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
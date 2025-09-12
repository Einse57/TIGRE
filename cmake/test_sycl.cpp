#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        sycl::queue q;
        std::cout << "SYCL device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SYCL test failed: " << e.what() << std::endl;
        return 1;
    }
}
# Hybrid build CMakeLists for SYCL + MSVC
cmake_minimum_required(VERSION 3.15)
project(TIGRE_SYCL_Backend LANGUAGES CXX)

# Set C++ standard  
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Force Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Option to enable SYCL backend
option(TIGRE_BUILD_SYCL_BACKEND "Enable SYCL backend compilation" ON)

message(STATUS "TIGRE_BUILD_SYCL_BACKEND: ${TIGRE_BUILD_SYCL_BACKEND}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files - factory only (no SYCL sources)
set(BACKEND_SOURCES
    src/backends/device_backend_factory_no_sycl.cpp
)

# Create library from factory only
add_library(tigre_backends ${BACKEND_SOURCES})

# Set compile definitions
if(TIGRE_BUILD_SYCL_BACKEND)
    target_compile_definitions(tigre_backends PRIVATE TIGRE_BUILD_SYCL_BACKEND=1)
    
    # Add SYCL include directory for headers
    set(SYCL_INCLUDE_DIR "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/include")
    target_include_directories(tigre_backends PRIVATE ${SYCL_INCLUDE_DIR})
    
    # Link with pre-compiled SYCL library
    target_link_libraries(tigre_backends PRIVATE 
        ${CMAKE_SOURCE_DIR}/tigre_sycl.lib
    )
    
    # SYCL runtime library - make PUBLIC so test executable can find it
    set(SYCL_LIB_DIR "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib")
    target_link_directories(tigre_backends PUBLIC ${SYCL_LIB_DIR})
    target_link_libraries(tigre_backends PUBLIC sycl8 libirc libircmd)
    
    message(STATUS "TIGRE SYCL backend enabled with pre-compiled library")
else()
    target_compile_definitions(tigre_backends PRIVATE TIGRE_BUILD_SYCL_BACKEND=0)
    message(STATUS "TIGRE SYCL backend disabled")
endif()

# Create test executable
add_executable(test_backend_factory
    tests/test_backend_factory.cpp
)
target_link_libraries(test_backend_factory tigre_backends)
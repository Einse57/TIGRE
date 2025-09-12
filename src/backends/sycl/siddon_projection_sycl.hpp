#pragma once

#ifdef TIGRE_BUILD_SYCL_BACKEND
#include <sycl/sycl.hpp>
#endif

#include "tigre/backend/device_backend.hpp"

namespace tigre {
namespace sycl_kernels {

#ifdef TIGRE_BUILD_SYCL_BACKEND

/**
 * @brief 3D Point structure compatible with CUDA version
 */
struct Point3D {
    float x, y, z;
    
    Point3D() = default;
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }
    
    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }
    
    Point3D operator*(float scalar) const {
        return Point3D(x * scalar, y * scalar, z * scalar);
    }
};

/**
 * @brief SYCL implementation of Siddon ray-voxel intersection algorithm
 * 
 * This kernel performs forward projection using the Siddon algorithm,
 * which efficiently computes ray-voxel intersections for tomographic
 * reconstruction.
 * 
 * Based on the original CUDA implementation in Siddon_projection.cu
 */
class SiddonProjectionKernel {
private:
    static constexpr int PROJ_PER_BLOCK = 9;
    static constexpr int PIXEL_SIZE_BLOCK = 9;
    static constexpr float EPS = 0.001f;

public:
    /**
     * @brief Launch Siddon projection kernel
     * @param queue SYCL queue for kernel execution
     * @param geo Geometry parameters
     * @param views View parameters (angles, offsets)
     * @param volume_data Input volume data on device
     * @param projection_data Output projection data on device
     * @param num_angles Number of projection angles
     */
    static void launch(
        sycl::queue& queue,
        const Geometry& geo,
        const Views& views,
        const float* volume_data,
        float* projection_data,
        int num_angles
    );

private:
    /**
     * @brief Main projection kernel function
     */
    static void kernel_pixel_detector(
        sycl::nd_item<3> item,
        const Geometry geo,
        float* detector,
        const int curr_proj_set_number,
        const int total_num_projections,
        const float* volume_data,
        const Point3D* proj_params_array
    );
    
    /**
     * @brief Trilinear interpolation for volume sampling (replaces CUDA tex3D)
     */
    static float sample_volume(
        const float* volume_data,
        const Geometry& geo,
        float x, float y, float z
    );
    
    /**
     * @brief Compute projection parameters for given angle
     */
    static void compute_projection_params(
        const Geometry& geo,
        float angle,
        float off_u,
        float off_v,
        Point3D& uv_origin,
        Point3D& delta_u,
        Point3D& delta_v,
        Point3D& source
    );
};

#endif // TIGRE_BUILD_SYCL_BACKEND

} // namespace sycl_kernels
} // namespace tigre
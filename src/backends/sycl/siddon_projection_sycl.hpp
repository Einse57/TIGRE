#ifndef SIDDON_PROJECTION_SYCL_FIXED_HPP
#define SIDDON_PROJECTION_SYCL_FIXED_HPP

#include "tigre/backend/device_backend.hpp"

#ifdef TIGRE_BUILD_SYCL_BACKEND
#include <sycl/sycl.hpp>
#endif

namespace tigre {
namespace sycl_kernels {

#ifdef TIGRE_BUILD_SYCL_BACKEND

constexpr int PROJ_PER_BLOCK = 4; // Reduced for safer memory usage

struct Point3D {
    float x, y, z;
    Point3D() : x(0), y(0), z(0) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

class SiddonProjectionKernel {
public:
    static void launch(
        sycl::queue& queue,
        const Geometry& geo,
        const Views& views,
        const float* volume_data,
        float* projection_data,
        int num_angles);

private:
    static void kernel_pixel_detector_safe(
        sycl::nd_item<3> item,
        const Geometry geo,
        float* detector,
        const int curr_proj_set_number,
        const int total_num_projections,
        const float* volume_data,
        const Point3D* proj_params_array);
        
    static void compute_projection_params(
        const Geometry& geo,
        float angle,
        float off_u,
        float off_v,
        Point3D& uv_origin,
        Point3D& delta_u,
        Point3D& delta_v,
        Point3D& source);
};

#endif // TIGRE_BUILD_SYCL_BACKEND

} // namespace sycl_kernels
} // namespace tigre

#endif // SIDDON_PROJECTION_SYCL_FIXED_HPP
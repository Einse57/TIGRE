#pragma once

#ifdef TIGRE_BUILD_SYCL_BACKEND
#include <sycl/sycl.hpp>
#endif

#include "tigre/backend/device_backend.hpp"
#include "siddon_projection_sycl.hpp"

namespace tigre {
namespace sycl_kernels {

#ifdef TIGRE_BUILD_SYCL_BACKEND

/**
 * @brief SYCL implementation of FDK backprojection algorithm
 * 
 * This kernel performs FDK (Feldkamp-Davis-Kress) backprojection
 * for cone-beam CT reconstruction.
 * 
 * Based on the original CUDA implementation in voxel_backprojection.cu
 */
class FDKBackprojectionKernel {
private:
    static constexpr int PROJ_PER_KERNEL = 32;
    static constexpr int VOXELS_PER_THREAD = 8;

public:
    /**
     * @brief Launch FDK backprojection kernel
     * @param queue SYCL queue for kernel execution
     * @param geo Geometry parameters
     * @param views View parameters (angles, offsets)
     * @param projection_data Input projection data on device
     * @param volume_data Output volume data on device (accumulated)
     * @param num_angles Number of projection angles
     */
    static void launch(
        sycl::queue& queue,
        const Geometry& geo,
        const Views& views,
        const float* projection_data,
        float* volume_data,
        int num_angles
    );

private:
    /**
     * @brief Main FDK backprojection kernel function
     */
    static void kernel_pixel_backprojection_fdk(
        sycl::nd_item<1> item,
        const Geometry geo,
        float* image,
        const int curr_proj_set_number,
        const int total_num_projections,
        const float* projection_data,
        const Point3D* proj_params_array,
        const float* proj_sin_cos_array
    );
    
    /**
     * @brief Sample projection data with bilinear interpolation
     */
    static float sample_projection(
        const float* projection_data,
        const Geometry& geo,
        float u, float v,
        int proj_idx
    );
    
    /**
     * @brief Compute backprojection parameters for given angle
     */
    static void compute_backprojection_params(
        const Geometry& geo,
        float angle,
        float off_u,
        float off_v,
        Point3D& delta_x,
        Point3D& delta_y,
        Point3D& delta_z,
        Point3D& xyz_origin,
        Point3D& off_orig,
        Point3D& off_detec,
        float& sin_alpha,
        float& cos_alpha
    );
};

#endif // TIGRE_BUILD_SYCL_BACKEND

} // namespace sycl_kernels
} // namespace tigre
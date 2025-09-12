#include "siddon_projection_sycl.hpp"
#include <cmath>
#include <algorithm>

#ifdef TIGRE_BUILD_SYCL_BACKEND
#include <sycl/sycl.hpp>
#endif

namespace tigre {
namespace sycl_kernels {

#ifdef TIGRE_BUILD_SYCL_BACKEND

void SiddonProjectionKernel::launch(
    sycl::queue& queue,
    const Geometry& geo,
    const Views& views,
    const float* volume_data,
    float* projection_data,
    int num_angles) {
    
    // Allocate device memory for projection parameters (equivalent to CUDA constant memory)
    size_t proj_params_size = 4 * PROJ_PER_BLOCK * sizeof(Point3D);
    Point3D* d_proj_params = sycl::malloc_device<Point3D>(4 * PROJ_PER_BLOCK, queue);
    
    // Process projections in blocks
    int num_kernel_calls = (num_angles + PROJ_PER_BLOCK - 1) / PROJ_PER_BLOCK;
    
    for (int kernel_call = 0; kernel_call < num_kernel_calls; kernel_call++) {
        int proj_start = kernel_call * PROJ_PER_BLOCK;
        int proj_end = std::min(proj_start + PROJ_PER_BLOCK, num_angles);
        int proj_this_block = proj_end - proj_start;
        
        // Prepare projection parameters for this block
        std::vector<Point3D> host_proj_params(4 * PROJ_PER_BLOCK);
        
        for (int proj = 0; proj < proj_this_block; proj++) {
            int angle_idx = proj_start + proj;
            Point3D uv_origin, delta_u, delta_v, source;
            
            float angle = views.theta[angle_idx];
            float off_u = (views.off_u.size() > angle_idx) ? views.off_u[angle_idx] : 0.0f;
            float off_v = (views.off_v.size() > angle_idx) ? views.off_v[angle_idx] : 0.0f;
            
            compute_projection_params(geo, angle, off_u, off_v, 
                                    uv_origin, delta_u, delta_v, source);
            
            host_proj_params[4 * proj + 0] = uv_origin;
            host_proj_params[4 * proj + 1] = delta_u;
            host_proj_params[4 * proj + 2] = delta_v;
            host_proj_params[4 * proj + 3] = source;
        }
        
        // Copy projection parameters to device
        queue.memcpy(d_proj_params, host_proj_params.data(), proj_params_size).wait();
        
        // Launch kernel
        sycl::range<3> global_range(proj_this_block, 
                                  (geo.nDetecU + PIXEL_SIZE_BLOCK - 1) / PIXEL_SIZE_BLOCK * PIXEL_SIZE_BLOCK,
                                  (geo.nDetecV + PIXEL_SIZE_BLOCK - 1) / PIXEL_SIZE_BLOCK * PIXEL_SIZE_BLOCK);
        sycl::range<3> local_range(1, PIXEL_SIZE_BLOCK, PIXEL_SIZE_BLOCK);
        
        queue.submit([&](sycl::handler& h) {
            h.parallel_for<class SiddonProjection>(
                sycl::nd_range<3>(global_range, local_range),
                [=](sycl::nd_item<3> item) {
                    kernel_pixel_detector(item, geo, projection_data, kernel_call,
                                        num_angles, volume_data, d_proj_params);
                });
        });
        
        queue.wait(); // Synchronize after each kernel call
    }
    
    // Cleanup
    sycl::free(d_proj_params, queue);
}

void SiddonProjectionKernel::kernel_pixel_detector(
    sycl::nd_item<3> item,
    const Geometry geo,
    float* detector,
    const int curr_proj_set_number,
    const int total_num_projections,
    const float* volume_data,
    const Point3D* proj_params_array) {
    
    // Get thread indices
    unsigned long u = item.get_global_id(1);
    unsigned long v = item.get_global_id(2);
    unsigned long proj_number = item.get_global_id(0);
    
    if (u >= geo.nDetecU || v >= geo.nDetecV || proj_number >= PROJ_PER_BLOCK) {
        return;
    }
    
    // Calculate linear detector index (matching CUDA version for Python compatibility)
#if IS_FOR_MATLAB_TIGRE
    size_t idx = (size_t)(u * (unsigned long long)geo.nDetecV + v) + 
                proj_number * (unsigned long long)geo.nDetecV * (unsigned long long)geo.nDetecU;
#else
    size_t idx = (size_t)(v * (unsigned long long)geo.nDetecU + u) + 
                proj_number * (unsigned long long)geo.nDetecV * (unsigned long long)geo.nDetecU;
#endif
    
    unsigned long ind_alpha = curr_proj_set_number * PROJ_PER_BLOCK + proj_number;
    
    if (ind_alpha >= total_num_projections) {
        return;
    }
    
    // Get projection parameters from constant memory equivalent
    Point3D uv_origin = proj_params_array[4 * proj_number + 0];
    Point3D delta_u = proj_params_array[4 * proj_number + 1];
    Point3D delta_v = proj_params_array[4 * proj_number + 2];
    Point3D source = proj_params_array[4 * proj_number + 3];
    
    // Get coordinates XYZ of pixel UV
    unsigned long pixel_v = geo.nDetecV - v - 1;
    unsigned long pixel_u = u;
    
    Point3D pixel_1d(
        uv_origin.x + pixel_u * delta_u.x + pixel_v * delta_v.x,
        uv_origin.y + pixel_u * delta_u.y + pixel_v * delta_v.y,
        uv_origin.z + pixel_u * delta_u.z + pixel_v * delta_v.z
    );
    
    // Siddon's ray-voxel intersection algorithm
    Point3D ray = pixel_1d - source;
    
    // Handle near-zero ray components
    ray.x = (sycl::fabs(ray.x) < EPS) ? 0.0f : ray.x;
    ray.y = (sycl::fabs(ray.y) < EPS) ? 0.0f : ray.y;
    ray.z = (sycl::fabs(ray.z) < EPS) ? 0.0f : ray.z;
    
    // Compute parameter values for x-ray parametric equation
    float axm = sycl::fmin((-source.x) / ray.x, (geo.nVoxelX - source.x) / ray.x);
    float aym = sycl::fmin((-source.y) / ray.y, (geo.nVoxelY - source.y) / ray.y);
    float azm = sycl::fmin((-source.z) / ray.z, (geo.nVoxelZ - source.z) / ray.z);
    float axM = sycl::fmax((-source.x) / ray.x, (geo.nVoxelX - source.x) / ray.x);
    float ayM = sycl::fmax((-source.y) / ray.y, (geo.nVoxelY - source.y) / ray.y);
    float azM = sycl::fmax((-source.z) / ray.z, (geo.nVoxelZ - source.z) / ray.z);
    
    float am = sycl::fmax(sycl::fmax(axm, aym), azm);
    float aM = sycl::fmin(sycl::fmin(axM, ayM), azM);
    
    // Check if line intersects voxel space
    if (am >= aM) {
        detector[idx] = 0.0f;
        return;
    }
    
    // Compute min/max image indices for intersection
    float imin, imax, jmin, jmax, kmin, kmax;
    
    // For X
    if (source.x < pixel_1d.x) {
        imin = (am == axm) ? 1.0f : sycl::ceil(source.x + am * ray.x);
        imax = (aM == axM) ? geo.nVoxelX : sycl::floor(source.x + aM * ray.x);
    } else {
        imax = (am == axm) ? geo.nVoxelX - 1.0f : sycl::floor(source.x + am * ray.x);
        imin = (aM == axM) ? 0.0f : sycl::ceil(source.x + aM * ray.x);
    }
    
    // For Y
    if (source.y < pixel_1d.y) {
        jmin = (am == aym) ? 1.0f : sycl::ceil(source.y + am * ray.y);
        jmax = (aM == ayM) ? geo.nVoxelY : sycl::floor(source.y + aM * ray.y);
    } else {
        jmax = (am == aym) ? geo.nVoxelY - 1.0f : sycl::floor(source.y + am * ray.y);
        jmin = (aM == ayM) ? 0.0f : sycl::ceil(source.y + aM * ray.y);
    }
    
    // For Z
    if (source.z < pixel_1d.z) {
        kmin = (am == azm) ? 1.0f : sycl::ceil(source.z + am * ray.z);
        kmax = (aM == azM) ? geo.nVoxelZ : sycl::floor(source.z + aM * ray.z);
    } else {
        kmax = (am == azm) ? geo.nVoxelZ - 1.0f : sycl::floor(source.z + am * ray.z);
        kmin = (aM == azM) ? 0.0f : sycl::ceil(source.z + aM * ray.z);
    }
    
    // Get intersection point parameters
    float ax = (source.x < pixel_1d.x) ? (imin - source.x) / ray.x : (imax - source.x) / ray.x;
    float ay = (source.y < pixel_1d.y) ? (jmin - source.y) / ray.y : (jmax - source.y) / ray.y;
    float az = (source.z < pixel_1d.z) ? (kmin - source.z) / ray.z : (kmax - source.z) / ray.z;
    
    // Handle infinite values (ray parallel to axis)
    ax = sycl::isinf(ax) ? sycl::fabs(ax) : ax;
    ay = sycl::isinf(ay) ? sycl::fabs(ay) : ay;
    az = sycl::isinf(az) ? sycl::fabs(az) : az;
    
    // Get index of first intersection
    float aminc = sycl::fmin(sycl::fmin(ax, ay), az);
    unsigned long i = (unsigned long)sycl::floor(source.x + (aminc + am) * 0.5f * ray.x);
    unsigned long j = (unsigned long)sycl::floor(source.y + (aminc + am) * 0.5f * ray.y);
    unsigned long k = (unsigned long)sycl::floor(source.z + (aminc + am) * 0.5f * ray.z);
    
    // Initialize traversal
    float ac = am;
    
    // Unit step sizes (reciprocals of ray direction)
    float axu = 1.0f / sycl::fabs(ray.x);
    float ayu = 1.0f / sycl::fabs(ray.y);
    float azu = 1.0f / sycl::fabs(ray.z);
    
    // Direction of update
    float iu = (source.x < pixel_1d.x) ? 1.0f : -1.0f;
    float ju = (source.y < pixel_1d.y) ? 1.0f : -1.0f;
    float ku = (source.z < pixel_1d.z) ? 1.0f : -1.0f;
    
    // Maximum ray length for scaling
    float max_length = sycl::sqrt(ray.x * ray.x * geo.dVoxelX * geo.dVoxelX +
                                 ray.y * ray.y * geo.dVoxelY * geo.dVoxelY +
                                 ray.z * ray.z * geo.dVoxelZ * geo.dVoxelZ);
    
    float sum = 0.0f;
    unsigned long Np = (imax - imin + 1) + (jmax - jmin + 1) + (kmax - kmin + 1);
    
    // Traverse ray through voxel grid
    i += 0.5f;
    j += 0.5f;
    k += 0.5f;
    
    for (unsigned long ii = 0; ii < Np; ii++) {
        // Sample volume at current position (replaces tex3D)
        float voxel_value = sample_volume(volume_data, geo, i, j, k);
        
        if (ax == aminc) {
            sum += (ax - ac) * voxel_value;
            i = i + iu;
            ac = ax;
            ax += axu;
        } else if (ay == aminc) {
            sum += (ay - ac) * voxel_value;
            j = j + ju;
            ac = ay;
            ay += ayu;
        } else if (az == aminc) {
            sum += (az - ac) * voxel_value;
            k = k + ku;
            ac = az;
            az += azu;
        }
        
        aminc = sycl::fmin(sycl::fmin(ax, ay), az);
    }
    
    detector[idx] = sum * max_length;
}

float SiddonProjectionKernel::sample_volume(
    const float* volume_data,
    const Geometry& geo,
    float x, float y, float z) {
    
    // Bounds checking
    if (x < 0 || x >= geo.nVoxelX || y < 0 || y >= geo.nVoxelY || z < 0 || z >= geo.nVoxelZ) {
        return 0.0f;
    }
    
    // For now, use nearest neighbor sampling (can be improved to trilinear)
    int ix = (int)sycl::floor(x);
    int iy = (int)sycl::floor(y);
    int iz = (int)sycl::floor(z);
    
    // Clamp to valid range
    ix = sycl::clamp(ix, 0, (int)geo.nVoxelX - 1);
    iy = sycl::clamp(iy, 0, (int)geo.nVoxelY - 1);
    iz = sycl::clamp(iz, 0, (int)geo.nVoxelZ - 1);
    
    // Compute linear index
    size_t idx = iz * geo.nVoxelY * geo.nVoxelX + iy * geo.nVoxelX + ix;
    
    return volume_data[idx];
}

void SiddonProjectionKernel::compute_projection_params(
    const Geometry& geo,
    float angle,
    float off_u,
    float off_v,
    Point3D& uv_origin,
    Point3D& delta_u,
    Point3D& delta_v,
    Point3D& source) {
    
    // This is a simplified version - full implementation would include
    // all the geometric transformations from the CUDA version
    float cos_alpha = sycl::cos(angle);
    float sin_alpha = sycl::sin(angle);
    
    // Source position (simplified - assumes circular orbit)
    source.x = geo.DSO[0] * sin_alpha;
    source.y = -geo.DSO[0] * cos_alpha;
    source.z = 0.0f;
    
    // Detector center
    Point3D detector_center;
    detector_center.x = (geo.DSD[0] - geo.DSO[0]) * sin_alpha;
    detector_center.y = -(geo.DSD[0] - geo.DSO[0]) * cos_alpha;
    detector_center.z = 0.0f;
    
    // UV coordinate system on detector
    // U direction (horizontal)
    delta_u.x = geo.dDetecU * cos_alpha;
    delta_u.y = geo.dDetecU * sin_alpha;
    delta_u.z = 0.0f;
    
    // V direction (vertical)
    delta_v.x = 0.0f;
    delta_v.y = 0.0f;
    delta_v.z = geo.dDetecV;
    
    // UV origin (corner of detector)
    float half_u = geo.nDetecU * 0.5f;
    float half_v = geo.nDetecV * 0.5f;
    
    uv_origin.x = detector_center.x - half_u * delta_u.x - half_v * delta_v.x;
    uv_origin.y = detector_center.y - half_u * delta_u.y - half_v * delta_v.y;
    uv_origin.z = detector_center.z - half_u * delta_u.z - half_v * delta_v.z;
    
    // Apply offsets
    uv_origin.x += off_u * delta_u.x + off_v * delta_v.x;
    uv_origin.y += off_u * delta_u.y + off_v * delta_v.y;
    uv_origin.z += off_u * delta_u.z + off_v * delta_v.z;
}

#endif // TIGRE_BUILD_SYCL_BACKEND

} // namespace sycl_kernels
} // namespace tigre
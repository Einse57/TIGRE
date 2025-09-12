#include "fdk_backprojection_sycl.hpp"
#include <cmath>
#include <algorithm>

#ifdef TIGRE_BUILD_SYCL_BACKEND
#include <sycl/sycl.hpp>
#endif

namespace tigre {
namespace sycl_kernels {

#ifdef TIGRE_BUILD_SYCL_BACKEND

void FDKBackprojectionKernel::launch(
    sycl::queue& queue,
    const Geometry& geo,
    const Views& views,
    const float* projection_data,
    float* volume_data,
    int num_angles) {
    
    // Allocate device memory for projection parameters
    size_t proj_params_size = 6 * PROJ_PER_KERNEL * sizeof(Point3D);
    size_t sin_cos_size = 5 * PROJ_PER_KERNEL * sizeof(float);
    
    Point3D* d_proj_params = sycl::malloc_device<Point3D>(6 * PROJ_PER_KERNEL, queue);
    float* d_proj_sin_cos = sycl::malloc_device<float>(5 * PROJ_PER_KERNEL, queue);
    
    // Calculate total number of voxels and work groups
    size_t total_voxels = geo.nVoxelX * geo.nVoxelY * geo.nVoxelZ;
    size_t local_size = 256; // Threads per work group
    size_t global_size = ((total_voxels + VOXELS_PER_THREAD - 1) / VOXELS_PER_THREAD + local_size - 1) / local_size * local_size;
    
    // Process projections in blocks
    int num_kernel_calls = (num_angles + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;
    
    for (int kernel_call = 0; kernel_call < num_kernel_calls; kernel_call++) {
        int proj_start = kernel_call * PROJ_PER_KERNEL;
        int proj_end = std::min(proj_start + PROJ_PER_KERNEL, num_angles);
        int proj_this_block = proj_end - proj_start;
        
        // Prepare projection parameters for this block
        std::vector<Point3D> host_proj_params(6 * PROJ_PER_KERNEL);
        std::vector<float> host_sin_cos(5 * PROJ_PER_KERNEL);
        
        for (int proj = 0; proj < proj_this_block; proj++) {
            int angle_idx = proj_start + proj;
            float angle = views.theta[angle_idx];
            float off_u = (views.off_u.size() > angle_idx) ? views.off_u[angle_idx] : 0.0f;
            float off_v = (views.off_v.size() > angle_idx) ? views.off_v[angle_idx] : 0.0f;
            
            Point3D delta_x, delta_y, delta_z, xyz_origin, off_orig, off_detec;
            float sin_alpha, cos_alpha;
            
            compute_backprojection_params(geo, angle, off_u, off_v,
                                        delta_x, delta_y, delta_z, xyz_origin,
                                        off_orig, off_detec, sin_alpha, cos_alpha);
            
            host_proj_params[6 * proj + 0] = delta_x;
            host_proj_params[6 * proj + 1] = delta_y;
            host_proj_params[6 * proj + 2] = delta_z;
            host_proj_params[6 * proj + 3] = xyz_origin;
            host_proj_params[6 * proj + 4] = off_orig;
            host_proj_params[6 * proj + 5] = off_detec;
            
            host_sin_cos[5 * proj + 0] = sin_alpha;
            host_sin_cos[5 * proj + 1] = cos_alpha;
            // Additional parameters can be added here if needed
        }
        
        // Copy parameters to device
        queue.memcpy(d_proj_params, host_proj_params.data(), proj_params_size).wait();
        queue.memcpy(d_proj_sin_cos, host_sin_cos.data(), sin_cos_size).wait();
        
        // Launch kernel
        queue.submit([&](sycl::handler& h) {
            h.parallel_for<class FDKBackprojection>(
                sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                [=](sycl::nd_item<1> item) {
                    kernel_pixel_backprojection_fdk(item, geo, volume_data, kernel_call,
                                                  num_angles, projection_data, 
                                                  d_proj_params, d_proj_sin_cos);
                });
        });
        
        queue.wait(); // Synchronize after each kernel call
    }
    
    // Cleanup
    sycl::free(d_proj_params, queue);
    sycl::free(d_proj_sin_cos, queue);
}

void FDKBackprojectionKernel::kernel_pixel_backprojection_fdk(
    sycl::nd_item<1> item,
    const Geometry geo,
    float* image,
    const int curr_proj_set_number,
    const int total_num_projections,
    const float* projection_data,
    const Point3D* proj_params_array,
    const float* proj_sin_cos_array) {
    
    // Get thread index and calculate voxel indices
    size_t thread_id = item.get_global_id(0);
    size_t voxels_per_thread = VOXELS_PER_THREAD;
    size_t start_voxel = thread_id * voxels_per_thread;
    
    size_t total_voxels = geo.nVoxelX * geo.nVoxelY * geo.nVoxelZ;
    size_t end_voxel = sycl::min(start_voxel + voxels_per_thread, total_voxels);
    
    if (start_voxel >= total_voxels) {
        return;
    }
    
    // Process multiple voxels per thread
    for (size_t voxel_idx = start_voxel; voxel_idx < end_voxel; voxel_idx++) {
        // Convert linear voxel index to 3D coordinates
        int k = voxel_idx / (geo.nVoxelX * geo.nVoxelY);
        int j = (voxel_idx % (geo.nVoxelX * geo.nVoxelY)) / geo.nVoxelX;
        int i = voxel_idx % geo.nVoxelX;
        
        // Physical voxel position
        float voxel_x = (i + 0.5f) * geo.dVoxelX - geo.sVoxelX / 2.0f;
        float voxel_y = (j + 0.5f) * geo.dVoxelY - geo.sVoxelY / 2.0f;
        float voxel_z = (k + 0.5f) * geo.dVoxelZ - geo.sVoxelZ / 2.0f;
        
        float voxel_value = 0.0f;
        
        // Iterate through all projections in this kernel call
        int proj_this_block = sycl::min(PROJ_PER_KERNEL, 
                                       total_num_projections - curr_proj_set_number * PROJ_PER_KERNEL);
        
        for (int proj = 0; proj < proj_this_block; proj++) {
            // Get projection parameters
            Point3D delta_x = proj_params_array[6 * proj + 0];
            Point3D delta_y = proj_params_array[6 * proj + 1];
            Point3D delta_z = proj_params_array[6 * proj + 2];
            Point3D xyz_origin = proj_params_array[6 * proj + 3];
            Point3D off_orig = proj_params_array[6 * proj + 4];
            Point3D off_detec = proj_params_array[6 * proj + 5];
            
            float sin_alpha = proj_sin_cos_array[5 * proj + 0];
            float cos_alpha = proj_sin_cos_array[5 * proj + 1];
            
            // Apply coordinate transformations
            Point3D world_pos;
            world_pos.x = voxel_x + off_orig.x;
            world_pos.y = voxel_y + off_orig.y;
            world_pos.z = voxel_z + off_orig.z;
            
            // Transform to projection coordinate system
            float proj_x = xyz_origin.x + world_pos.x * delta_x.x + world_pos.y * delta_y.x + world_pos.z * delta_z.x;
            float proj_y = xyz_origin.y + world_pos.x * delta_x.y + world_pos.y * delta_y.y + world_pos.z * delta_z.y;
            float proj_z = xyz_origin.z + world_pos.x * delta_x.z + world_pos.y * delta_y.z + world_pos.z * delta_z.z;
            
            // Avoid division by zero
            if (sycl::fabs(proj_z) < 1e-6f) {
                continue;
            }
            
            // Calculate detector coordinates
            float u = proj_x / proj_z;
            float v = proj_y / proj_z;
            
            // Apply detector offsets
            u += off_detec.x;
            v += off_detec.y;
            
            // Convert to detector pixel coordinates
            float det_u = (u + geo.sDetecU / 2.0f) / geo.dDetecU;
            float det_v = (v + geo.sDetecV / 2.0f) / geo.dDetecV;
            
            // Check if projection coordinates are within detector bounds
            if (det_u >= 0.0f && det_u < geo.nDetecU && det_v >= 0.0f && det_v < geo.nDetecV) {
                // Sample projection data with bilinear interpolation
                int proj_idx = curr_proj_set_number * PROJ_PER_KERNEL + proj;
                float proj_value = sample_projection(projection_data, geo, det_u, det_v, proj_idx);
                
                // FDK weighting
                float weight = geo.DSO[proj_idx] / sycl::sqrt(geo.DSO[proj_idx] * geo.DSO[proj_idx] + 
                                                              u * u + v * v);
                
                voxel_value += proj_value * weight;
            }
        }
        
        // Accumulate to output volume (atomic add for thread safety)
        // Note: SYCL atomic operations may have different syntax
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> 
            atomic_voxel(image[voxel_idx]);
        atomic_voxel += voxel_value;
    }
}

float FDKBackprojectionKernel::sample_projection(
    const float* projection_data,
    const Geometry& geo,
    float u, float v,
    int proj_idx) {
    
    // Bounds checking
    if (u < 0.0f || u >= geo.nDetecU - 1 || v < 0.0f || v >= geo.nDetecV - 1) {
        return 0.0f;
    }
    
    // Bilinear interpolation
    int u_low = (int)sycl::floor(u);
    int v_low = (int)sycl::floor(v);
    int u_high = u_low + 1;
    int v_high = v_low + 1;
    
    float u_frac = u - u_low;
    float v_frac = v - v_low;
    
    // Calculate linear indices for projection data
    size_t proj_offset = proj_idx * geo.nDetecU * geo.nDetecV;
    
    size_t idx_00 = proj_offset + v_low * geo.nDetecU + u_low;
    size_t idx_01 = proj_offset + v_low * geo.nDetecU + u_high;
    size_t idx_10 = proj_offset + v_high * geo.nDetecU + u_low;
    size_t idx_11 = proj_offset + v_high * geo.nDetecU + u_high;
    
    // Bilinear interpolation
    float val_0 = projection_data[idx_00] * (1.0f - u_frac) + projection_data[idx_01] * u_frac;
    float val_1 = projection_data[idx_10] * (1.0f - u_frac) + projection_data[idx_11] * u_frac;
    
    return val_0 * (1.0f - v_frac) + val_1 * v_frac;
}

void FDKBackprojectionKernel::compute_backprojection_params(
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
    float& cos_alpha) {
    
    // Compute trigonometric values
    sin_alpha = sycl::sin(angle);
    cos_alpha = sycl::cos(angle);
    
    // This is a simplified version - full implementation would include
    // all the geometric transformations from the CUDA version
    
    // Delta vectors for coordinate transformation
    delta_x.x = cos_alpha;
    delta_x.y = sin_alpha;
    delta_x.z = 0.0f;
    
    delta_y.x = -sin_alpha;
    delta_y.y = cos_alpha;
    delta_y.z = 0.0f;
    
    delta_z.x = 0.0f;
    delta_z.y = 0.0f;
    delta_z.z = 1.0f;
    
    // Origin in projection coordinate system
    xyz_origin.x = geo.DSO[0];
    xyz_origin.y = 0.0f;
    xyz_origin.z = 0.0f;
    
    // Offsets (simplified)
    off_orig.x = 0.0f;
    off_orig.y = 0.0f;
    off_orig.z = 0.0f;
    
    off_detec.x = off_u;
    off_detec.y = off_v;
    off_detec.z = 0.0f;
}

#endif // TIGRE_BUILD_SYCL_BACKEND

} // namespace sycl_kernels
} // namespace tigre
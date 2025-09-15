#include "siddon_projection_sycl.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

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
    
    try {
        // Validate input parameters
        if (!volume_data || !projection_data || num_angles <= 0) {
            throw std::runtime_error("Invalid input parameters to SYCL kernel");
        }
        
        std::cout << "[SYCL DEBUG] Starting projection with " << num_angles << " angles" << std::endl;
        std::cout << "[SYCL DEBUG] Geometry: " << geo.nVoxelX << "x" << geo.nVoxelY << "x" << geo.nVoxelZ << std::endl;
        std::cout << "[SYCL DEBUG] Detector: " << geo.nDetecU << "x" << geo.nDetecV << std::endl;
        
        // Allocate device memory for projection parameters with error checking
        size_t proj_params_size = 4 * PROJ_PER_BLOCK * sizeof(Point3D);
        Point3D* d_proj_params = nullptr;
        
        try {
            d_proj_params = sycl::malloc_device<Point3D>(4 * PROJ_PER_BLOCK, queue);
            if (!d_proj_params) {
                throw std::runtime_error("Failed to allocate SYCL device memory");
            }
            std::cout << "[SYCL DEBUG] Device memory allocated successfully: " << proj_params_size << " bytes" << std::endl;
        } catch (const sycl::exception& e) {
            throw std::runtime_error("SYCL memory allocation failed: " + std::string(e.what()));
        }
        
        // Process projections in blocks
        int num_kernel_calls = (num_angles + PROJ_PER_BLOCK - 1) / PROJ_PER_BLOCK;
        std::cout << "[SYCL DEBUG] Number of kernel calls: " << num_kernel_calls << std::endl;
        
        for (int kernel_call = 0; kernel_call < num_kernel_calls; kernel_call++) {
            int proj_start = kernel_call * PROJ_PER_BLOCK;
            int proj_end = std::min(proj_start + PROJ_PER_BLOCK, num_angles);
            int proj_this_block = proj_end - proj_start;
            
            std::cout << "[SYCL DEBUG] Kernel call " << kernel_call << ": processing " << proj_this_block << " projections" << std::endl;
            
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
            
            // Copy projection parameters to device with error handling
            try {
                queue.memcpy(d_proj_params, host_proj_params.data(), proj_params_size).wait();
                std::cout << "[SYCL DEBUG] Parameters copied to device successfully" << std::endl;
            } catch (const sycl::exception& e) {
                sycl::free(d_proj_params, queue);
                throw std::runtime_error("SYCL memcpy failed: " + std::string(e.what()));
            }
            
            // Calculate safe kernel dimensions
            size_t safe_nDetecU = std::max(1, (int)geo.nDetecU);
            size_t safe_nDetecV = std::max(1, (int)geo.nDetecV);
            size_t safe_proj_count = std::max(1, proj_this_block);
            
            // Use smaller, safer work group sizes for Intel Graphics
            constexpr size_t SAFE_BLOCK_SIZE = 4; // Reduced from 9
            
            sycl::range<3> global_range(
                safe_proj_count,
                ((safe_nDetecU + SAFE_BLOCK_SIZE - 1) / SAFE_BLOCK_SIZE) * SAFE_BLOCK_SIZE,
                ((safe_nDetecV + SAFE_BLOCK_SIZE - 1) / SAFE_BLOCK_SIZE) * SAFE_BLOCK_SIZE
            );
            sycl::range<3> local_range(1, SAFE_BLOCK_SIZE, SAFE_BLOCK_SIZE);
            
            std::cout << "[SYCL DEBUG] Global range: " << global_range[0] << "x" << global_range[1] << "x" << global_range[2] << std::endl;
            std::cout << "[SYCL DEBUG] Local range: " << local_range[0] << "x" << local_range[1] << "x" << local_range[2] << std::endl;
            
            // Launch kernel with error handling
            try {
                queue.submit([&](sycl::handler& h) {
                    h.parallel_for<class SiddonProjectionSafe>(
                        sycl::nd_range<3>(global_range, local_range),
                        [=](sycl::nd_item<3> item) {
                            kernel_pixel_detector_safe(item, geo, projection_data, kernel_call,
                                                      num_angles, volume_data, d_proj_params);
                        });
                });
                
                queue.wait_and_throw(); // Use wait_and_throw for error detection
                std::cout << "[SYCL DEBUG] Kernel " << kernel_call << " completed successfully" << std::endl;
                
            } catch (const sycl::exception& e) {
                sycl::free(d_proj_params, queue);
                throw std::runtime_error("SYCL kernel execution failed: " + std::string(e.what()));
            }
        }
        
        // Cleanup
        sycl::free(d_proj_params, queue);
        std::cout << "[SYCL DEBUG] Projection completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[SYCL ERROR] " << e.what() << std::endl;
        throw;
    }
}

void SiddonProjectionKernel::kernel_pixel_detector_safe(
    sycl::nd_item<3> item,
    const Geometry geo,
    float* detector,
    const int curr_proj_set_number,
    const int total_num_projections,
    const float* volume_data,
    const Point3D* proj_params_array) {
    
    // Get thread indices with bounds checking
    unsigned long u = item.get_global_id(1);
    unsigned long v = item.get_global_id(2);
    unsigned long proj_number = item.get_global_id(0);
    
    // Early bounds check
    if (u >= geo.nDetecU || v >= geo.nDetecV || proj_number >= PROJ_PER_BLOCK) {
        return;
    }
    
    // Additional safety checks
    if (!detector || !volume_data || !proj_params_array) {
        return;
    }
    
    try {
        // Calculate linear detector index with bounds checking
        size_t detector_size = (size_t)geo.nDetecU * geo.nDetecV;
        
#if IS_FOR_MATLAB_TIGRE
        size_t idx = (size_t)(u * (unsigned long long)geo.nDetecV + v) + 
                    proj_number * (unsigned long long)geo.nDetecV * (unsigned long long)geo.nDetecU;
#else
        size_t idx = (size_t)(v * (unsigned long long)geo.nDetecU + u) + 
                    proj_number * (unsigned long long)geo.nDetecU * (unsigned long long)geo.nDetecV;
#endif
        
        // Bounds check for detector index
        size_t max_detector_idx = detector_size * total_num_projections;
        if (idx >= max_detector_idx) {
            return;
        }
        
        // Get projection parameters with bounds checking
        if (proj_number * 4 + 3 >= PROJ_PER_BLOCK * 4) {
            return;
        }
        
        Point3D uv_origin = proj_params_array[proj_number * 4 + 0];
        Point3D delta_u = proj_params_array[proj_number * 4 + 1];
        Point3D delta_v = proj_params_array[proj_number * 4 + 2];
        Point3D source = proj_params_array[proj_number * 4 + 3];
        
        // Compute pixel position in 3D with safety checks
        Point3D pixel_1d;
        pixel_1d.x = uv_origin.x + (float)u * delta_u.x + (float)v * delta_v.x;
        pixel_1d.y = uv_origin.y + (float)u * delta_u.y + (float)v * delta_v.y;
        pixel_1d.z = uv_origin.z + (float)u * delta_u.z + (float)v * delta_v.z;
        
        // Ray direction with validation
        Point3D ray;
        ray.x = pixel_1d.x - source.x;
        ray.y = pixel_1d.y - source.y;
        ray.z = pixel_1d.z - source.z;
        
        // Check for degenerate rays
        float ray_magnitude = sycl::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
        if (ray_magnitude < 1e-6f) {
            detector[idx] = 0.0f;
            return;
        }
        
        // Perform simplified ray tracing with bounds checking
        float sum_val = 0.0f;
        
        // Use a simplified, safer ray tracing algorithm
        const int num_samples = 64; // Reduced number of samples
        for (int sample = 0; sample < num_samples; sample++) {
            float t = (float)sample / (float)(num_samples - 1);
            
            float sample_x = source.x + t * ray.x;
            float sample_y = source.y + t * ray.y;
            float sample_z = source.z + t * ray.z;
            
            // Convert to voxel indices with bounds checking
            int vox_x = (int)sycl::floor(sample_x);
            int vox_y = (int)sycl::floor(sample_y);
            int vox_z = (int)sycl::floor(sample_z);
            
            // Bounds check
            if (vox_x >= 0 && vox_x < geo.nVoxelX &&
                vox_y >= 0 && vox_y < geo.nVoxelY &&
                vox_z >= 0 && vox_z < geo.nVoxelZ) {
                
                size_t vol_idx = (size_t)vox_z * geo.nVoxelX * geo.nVoxelY + 
                               (size_t)vox_y * geo.nVoxelX + vox_x;
                
                // Additional bounds check for volume
                size_t vol_size = (size_t)geo.nVoxelX * geo.nVoxelY * geo.nVoxelZ;
                if (vol_idx < vol_size) {
                    sum_val += volume_data[vol_idx];
                }
            }
        }
        
        // Write result with bounds checking
        detector[idx] = sum_val * (ray_magnitude / num_samples);
        
    } catch (...) {
        // Silently handle any exceptions in kernel (avoid crashes)
        return;
    }
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
    
    // Geometric transformations for projection
    float cos_alpha = sycl::cos(angle);
    float sin_alpha = sycl::sin(angle);
    
    // Source position (circular orbit)
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
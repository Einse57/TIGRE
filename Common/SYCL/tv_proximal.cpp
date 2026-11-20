/*-------------------------------------------------------------------------
 *
 * MATLAB MEX  functions for TV image denoising. Check inputs and parses
 * MATLAB data to C++ data.
 *
 *
 * CODE by   Imanol Luengo
 *           PhD student University of Nottingham
 *           imaluengo@gmail.com
 *           2015
 *           Modified by Ander Biguri for multi-GPU 
 * ---------------------------------------------------------------------------
 * ---------------------------------------------------------------------------
 * Copyright (c) 2015, University of Bath and CERN- European Organization for
 * Nuclear Research
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------------
 *
 * Contact: tigre.toolbox@gmail.com
 * Codes  : https://github.com/CERN/TIGRE
 * ---------------------------------------------------------------------------
 */



// http://gpu4vision.icg.tugraz.at/papers/2010/knoll.pdf#pub47
#define MAXTREADS 1024
#define MAX_BUFFER 60
#define BLOCK_SIZE 10  // BLOCK_SIZE^3 must be smaller than MAXTREADS

#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include "tv_proximal.hpp"
#include "tigre_sycl.hpp"
#include "gpuUtils.hpp"
#include "tigre_release_on_error.hpp"

void cpy_from_host(float* device_array,float* host_array, 
                   unsigned long long bytes_device,unsigned long long offset_device,unsigned long long offset_host, 
                   unsigned long long pixels_per_slice, unsigned int buffer_length, 
                   sycl::queue* stream, bool is_first_chunk, bool is_last_chunk,const long* image_size);  
    
    
// The kernels of each TIGRE module share one SYCL runtime in a Python process, and the
// runtime finds kernels by name. A named namespace keeps these kernel names distinct
// from the identically named helpers of the other TV modules.
namespace tigre_tv_proximal {
    static inline
            float divergence(const float* pz, const float* py, const float* px,
            long z, long y, long x, long depth, long rows, long cols,
            float dz, float dy, float dx)
    {
        long size2d = rows*cols;
        size_t idx = (size_t)z * size2d + y * cols + x;
        float _div = 0.0f;
        
        if ( z - 1 >= 0 ) {
            _div += (pz[idx] - pz[(size_t)(z-1)*size2d + y*cols + x]) / dz;
        } else {
            _div += pz[idx];
        }
        
        if ( y - 1 >= 0 ) {
            _div += (py[idx] - py[(size_t)z*size2d + (y-1)*cols + x]) / dy;
        } else {
            _div += py[idx];
        }
        
        if ( x - 1 >= 0 ) {
            _div += (px[idx] - px[(size_t)z*size2d + y*cols + (x-1)]) / dx;
        } else {
            _div += px[idx];
        }
        
        return _div;
    }
    
    static inline
            void gradient(const float* u, float* grad,
            long z, long y, long x,
            long depth, long rows, long cols,
            float dz, float dy, float dx)
    {
        long size2d = rows*cols;
        size_t idx = (size_t)z * size2d + y * cols + x;
        
        float uidx = u[idx];
        
        if ( z + 1 < depth ) {
            grad[0] = (u[(size_t)(z+1)*size2d + y*cols + x] - uidx) / dz;
        }
        
        if ( y + 1 < rows ) {
            grad[1] = (u[(size_t)z*size2d + (y+1)*cols + x] - uidx) / dy;
        }
        
        if ( x + 1 < cols ) {
            grad[2] = (u[(size_t)z*size2d + y*cols + (x+1)] - uidx) / dx;
        }
    }
    
    
    static inline
            void update_u(const float* f, const float* pz, const float* py, const float* px, float* u,
            float tau, float lambda,
            long depth, long rows, long cols,
            float dz, float dy, float dx,
            long x, long y, long z)
    {
        size_t idx = (size_t)z * (rows * cols) + y * cols + x;
        
        if ( x >= cols || y >= rows || z >= depth )
            return;
        
        float _div = divergence(pz, py, px, z, y, x, depth, rows, cols, dz, dy, dx);
        
        u[idx] = u[idx] * (1.0f - tau) + tau * (f[idx] + (1.0f/lambda) * _div);
    }
    
    
    static inline
            void update_p(const float* u, float* pz, float* py, float* px,
            float tau, long depth, long rows, long cols,
            float dz, float dy, float dx,
            long x, long y, long z)
    {
        size_t idx = (size_t)z * (rows * cols) + y * cols + x;
        
        if ( x >= cols || y >= rows || z >= depth )
            return;
        
        float grad[3] = {0,0,0}, q[3];
        gradient(u, grad, z, y, x, depth, rows, cols, dz, dy, dx);
        
        q[0] = pz[idx] + tau * grad[0];
        q[1] = py[idx] + tau * grad[1];
        q[2] = px[idx] + tau * grad[2];
        
        float norm = sycl::fmax(1.0f, sycl::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]));
        
        pz[idx] = q[0] / norm;
        py[idx] = q[1] / norm;
        px[idx] = q[2] / norm;
    }
    
    
    static void update_u(sycl::queue& q, const float* f, const float* pz, const float* py, const float* px, float* u,
            float tau, float lambda,
            long depth, long rows, long cols,
            float dz, float dy, float dx)
    {
        q.parallel_for(sycl::range<3>(depth, rows, cols), [=](sycl::id<3> id){
            update_u(f, pz, py, px, u, tau, lambda, depth, rows, cols, dz, dy, dx, id[2], id[1], id[0]);
        });
    }
    static void update_p(sycl::queue& q, const float* u, float* pz, float* py, float* px,
            float tau, long depth, long rows, long cols,
            float dz, float dy, float dx)
    {
        q.parallel_for(sycl::range<3>(depth, rows, cols), [=](sycl::id<3> id){
            update_p(u, pz, py, px, tau, depth, rows, cols, dz, dy, dx, id[2], id[1], id[0]);
        });
    }
    
    
}  // namespace tigre_tv_proximal
using namespace tigre_tv_proximal;

// Main function
    void tvdenoising(float* src, float* dst, float lambda,
            const float* spacing, const long* image_size, int maxIter, const GpuIds& gpuids) {
    try {
        
            // Prepare for MultiGPU
            int deviceCount = gpuids.GetLength();
            if (deviceCount == 0) {
                mexErrMsgIdAndTxt("tvDenoise:tvdenoising:GPUselect","There are no available device(s) that support SYCL\n");
            }
            //
            // CODE assumes
            // 1.-All available devices are usable by this code
            // 2.-All available devices are equal, they are the same machine (warning thrown)
            // Check the available devices, and if they are the same
            if (!gpuids.AreEqualDevices()) {
                mexWarnMsgIdAndTxt("tvDenoise:tvdenoising:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
            }
            int dev;
            sycl::queue* q=&tigreSyclQueue(gpuids[0]);

            // We don't know if the devices are being used. lets check that. and only use the amount of memory we need.
        
            size_t mem_GPU_global;
            checkFreeMemory(gpuids, &mem_GPU_global);
        
        
            // %5 of free memory should be enough, we have almost no variables in these kernels
           size_t total_pixels           = (size_t)image_size[0]*image_size[1] * image_size[2] ;
           const size_t pixels_per_slice = (size_t)image_size[0]*image_size[1] ;
           const size_t mem_slice_image  = sizeof(float)* pixels_per_slice  ;
           const size_t mem_size_image   = sizeof(float)* total_pixels;
        
            // Decide how are we handling the distribution of computation
            size_t mem_img_each_GPU;
        
            unsigned int buffer_length=1;
            //Does everything fit in the GPU?
            unsigned int slices_per_split;
            unsigned int splits=1; // if the number does not fit in an uint, you have more serious trouble than this.
            if(mem_GPU_global> 5*mem_size_image+5*mem_slice_image*buffer_length*2){
                // We only need to split if we have extra GPUs
                slices_per_split=(image_size[2]+deviceCount-1)/deviceCount;
                // No empty pieces (see tvPiecesWithoutEmpty); this may use fewer GPUs or smaller pieces.
                tvPiecesWithoutEmpty((unsigned long)image_size[2], &deviceCount, &splits, &slices_per_split);
                mem_img_each_GPU=mem_slice_image*(slices_per_split + buffer_length*2);
            }else{
                // The smallest working set: on each GPU, 5 images of one slice plus the
                // minimum overlap (1 slice on each side). If not even that fits, say so
                // now: the sizes below are unsigned and would wrap around past zero.
                const size_t mem_min=5*mem_slice_image*(1+2*1);
                if (mem_GPU_global<=mem_min)
                    tigreNotEnoughGpuMemory("tvDenoise:tvdenoising:GPUmemory","Total variation denoising (tvdenoise)",mem_min,mem_GPU_global);
                // As mem_auxiliary is not expected to be a large value (for a 2000^3 image is around 28Mbytes), lets for now assume we need it all
                size_t mem_free=mem_GPU_global;
            
                splits=(unsigned int)(ceil(((float)(5*mem_size_image)/(float)(deviceCount))/mem_free));
                // Now, there is an overhead here, as each splits should have 2 slices more, to account for overlap of images.
                // lets make sure these 2 slices fit, if they do not, add 1 to splits.
                slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits);
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            
                // if the new stuff does not fit in the GPU, it means we are in the edge case where adding that extra slice will overflow memory
                // One more split usually does it; with very little memory it may take more
                // (one slice per piece always fits, checked above).
                while (mem_GPU_global< 5*mem_img_each_GPU && slices_per_split>1){
                    splits++;
                    //recompute for later
                    slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits); // amount of slices that fit on a GPU. Later we add 2 to these, as we need them for overlap
                    mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
                }
                // No empty pieces (see tvPiecesWithoutEmpty); this may use fewer GPUs or smaller pieces.
                tvPiecesWithoutEmpty((unsigned long)image_size[2], &deviceCount, &splits, &slices_per_split);
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            
                // How many EXTRA buffer slices should be able to fit in here??!?!
                mem_free=mem_GPU_global-(5*mem_img_each_GPU);
                unsigned int extra_buff=(mem_free/mem_slice_image);
                buffer_length=(extra_buff/2)/5; // we need double whatever this results in, rounded down.
                // At least 1: the main loop advances by buffer_length slices, so 0 never ended.
                if (buffer_length<1) buffer_length=1;
                buffer_length=std::min((unsigned int)MAX_BUFFER,buffer_length);
                // No wider than a piece: a piece's halo then comes from its neighbour piece only (see "carry" below).
                buffer_length=std::min(slices_per_split,buffer_length);
            
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            
                // Assert
                if (mem_GPU_global< 5*mem_img_each_GPU){
                    mexErrMsgIdAndTxt("tvDenoise:tvdenoising:GPU","Bad assert. Logic behind splitting flawed! Please tell: ander.biguri@gmail.com\n");
                }
            }
        
        
        
        
        
            // Lets allocate auxiliary  variables.
            // onError releases what the call holds if it is left by an error (see
            // tigre_release_on_error.hpp), as in the CUDA build (#870).
            tigre::ReleaseOnError onError;
            for (int d = 0; d < deviceCount; d++) onError.addQueue(&tigreSyclQueue(gpuids[d]));
            float* buffer_u, *buffer_px, *buffer_py, *buffer_pz;
            float* h_px, *h_py, *h_pz, *h_u;
            // With splits>1 the pieces run one after another and write their results back to the host. The lower halo
            // of the first piece of a split is the top of the last piece of the previous split, which has already been
            // advanced by then. "carry" keeps those slices (u, px, py, pz) as they were before that piece ran, so every
            // piece starts from neighbours at its own iteration count.
            float* carry=NULL;
            if(splits>1){
            
                //These take A LOT of memory and A LOT of time to use. If we can avoid using them, better.
                if (buffer_length<maxIter){ // if we do only 1 big iter, they are not needed.
                    mexWarnMsgIdAndTxt("tvDenoise:tvdenoising:Memory","TV denoising requires 5 times the image memory. Your GPU(s) do not have the required memory.\n This memory will be attempted to allocate on the CPU, which may fail or slow the computation by a very significant amount.\n If you want to kill the execution: CTRL+C");
                
                    h_px=(float*)malloc((size_t)image_size[0]*image_size[1]*image_size[2]*sizeof(float));
                    onError.add([p=h_px]{ free(p); });
                
                    h_py=(float*)malloc((size_t)image_size[0]*image_size[1]*image_size[2]*sizeof(float));
                    onError.add([p=h_py]{ free(p); });
                
                    h_pz=(float*)malloc((size_t)image_size[0]*image_size[1]*image_size[2]*sizeof(float));
                    onError.add([p=h_pz]{ free(p); });

                    carry=(float*)malloc(4*(size_t)buffer_length*pixels_per_slice*sizeof(float));
                    onError.add([p=carry]{ free(p); });
                    if (!h_px || !h_py || !h_pz || !carry)
                        mexErrMsgIdAndTxt("tvDenoise:tvdenoising:Memory","Malloc error on auxiliary variables on CPU.\n Your image is too big to use SART_TV or im3Ddenoise in your current machine");
                }
                h_u=dst;
            }else{
                // The exchange between devices copies buffer_length slices: with splits==1 from the
                // "does not fit" branch above, buffer_length can be more than 1.
                buffer_u=(float*)malloc(  (size_t)buffer_length*pixels_per_slice*sizeof(float));
                onError.add([p=buffer_u]{ free(p); });
                buffer_px=(float*)malloc( (size_t)buffer_length*pixels_per_slice*sizeof(float));
                onError.add([p=buffer_px]{ free(p); });
                buffer_py=(float*)malloc( (size_t)buffer_length*pixels_per_slice*sizeof(float));
                onError.add([p=buffer_py]{ free(p); });
                buffer_pz=(float*)malloc( (size_t)buffer_length*pixels_per_slice*sizeof(float));
                onError.add([p=buffer_pz]{ free(p); });
            
            }
            // We should be good to go memory wise.
        
        
            float** d_src   =(float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_src]{ free(p); });
            float** d_u     =(float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_u]{ free(p); });
            float** d_px    =(float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_px]{ free(p); });
            float** d_py    =(float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_py]{ free(p); });
            float** d_pz    =(float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_pz]{ free(p); });
        
            //Malloc
            for(dev=0;dev<deviceCount;dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                // F
                d_src[dev]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "tvDenoise:tvdenoising:GPU", "Total variation denoising (tvdenoise)");
                onError.add([p=d_src[dev], q]{ sycl::free(p, *q); });
                // U
                d_u [dev]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "tvDenoise:tvdenoising:GPU", "Total variation denoising (tvdenoise)");
                onError.add([p=d_u [dev], q]{ sycl::free(p, *q); });
                // PX
                d_px[dev]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "tvDenoise:tvdenoising:GPU", "Total variation denoising (tvdenoise)");
                onError.add([p=d_px[dev], q]{ sycl::free(p, *q); });
                // PY
                d_py[dev]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "tvDenoise:tvdenoising:GPU", "Total variation denoising (tvdenoise)");
                onError.add([p=d_py[dev], q]{ sycl::free(p, *q); });
                // PZ
                d_pz[dev]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "tvDenoise:tvdenoising:GPU", "Total variation denoising (tvdenoise)");
                onError.add([p=d_pz[dev], q]{ sycl::free(p, *q); });
            }
            q->wait();
        
        
            // Create streams
            int nStream_device=5;
            int nStreams=deviceCount*nStream_device;
            sycl::queue** stream=(sycl::queue**)malloc(nStreams*sizeof(sycl::queue*));
            onError.add([p=stream]{ free(p); });
        
            for (dev = 0; dev < deviceCount; dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                for (int i = 0; i < nStream_device; ++i){
                    stream[i+dev*nStream_device]=q;
                }
            }
        
        
        
        
            // Allocate CPU buffer if needed, warn user if not.
        
        
        
            unsigned int curr_slices;
            unsigned long long curr_pixels;
            size_t linear_idx_start;
        
            unsigned long long buffer_pixels=buffer_length*pixels_per_slice;
        
            unsigned long long* offset_device=(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
            onError.add([p=offset_device]{ free(p); });
            unsigned long long* offset_host  =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
            onError.add([p=offset_host]{ free(p); });
            unsigned long long* bytes_device =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
            onError.add([p=bytes_device]{ free(p); });
            // Halo slices below and above each piece: buffer_length, or fewer where the piece is closer than that to the
            // image's first or last slice. There the kernels' own boundary conditions apply, in z as in x and y, so the
            // result does not depend on how the image is split. On the device, a piece's interior always starts at
            // buffer_pixels; halo_lo[dev] slices below it and halo_hi[dev] above it are used.
            unsigned int* halo_lo=(unsigned int*)malloc(deviceCount*sizeof(unsigned int));
            onError.add([p=halo_lo]{ free(p); });
            unsigned int* halo_hi=(unsigned int*)malloc(deviceCount*sizeof(unsigned int));
            onError.add([p=halo_hi]{ free(p); });
            bool is_first_chunk;
            bool is_last_chunk;
        
            float tau2, tau1;

            for(unsigned int i=0;i<maxIter;i+=(buffer_length)){
           
                for(unsigned int sp=0;sp<splits;sp++){
                
                    // For each iteration we need to compute all the image. The ordering of these loops
                    // need to be like this due to the bounding layers between splits. If more than 1 split is needed
                    // for each GPU then there is no other way that taking the entire memory out of GPU and putting it back.
                    // If the memory can be shared between GPUs fully without extra splits, then there is an easy way of synchronizing the memory
                
                    // Copy image to memory
                    for (dev = 0; dev < deviceCount; dev++){
                        // Precompute indices and needed bytes
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        curr_pixels=curr_slices*pixels_per_slice;
                        linear_idx_start=pixels_per_slice*slices_per_split*(sp*deviceCount+dev);
                    
                        // Check if its the first or last chunck
                        is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                        is_first_chunk=!(sp*deviceCount+dev);
                    
                        // lets compute where we start copies and how much. This avoids 3 calls to Memcpy
                        size_t start_slice=(size_t)slices_per_split*(sp*deviceCount+dev);
                        halo_lo[dev]=(unsigned int)std::min((size_t)buffer_length, start_slice);
                        halo_hi[dev]=(unsigned int)std::min((size_t)buffer_length, (size_t)image_size[2]-start_slice-curr_slices);
                        offset_device[dev]=(buffer_length-halo_lo[dev])*pixels_per_slice;
                        offset_host[dev]=linear_idx_start-halo_lo[dev]*pixels_per_slice;
                        bytes_device[dev]=curr_pixels+(halo_lo[dev]+halo_hi[dev])*pixels_per_slice;
                    }
                    // copy data to the GPU if we are just starting
                    if(i==0){
                        for (dev = 0; dev < deviceCount; dev++){
                            is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                            is_first_chunk=!(sp*deviceCount+dev);

                            q=&tigreSyclQueue(gpuids[dev]);
                            if (is_last_chunk) {stream[dev*nStream_device+1]->memset(d_src[dev],0,mem_img_each_GPU);}
                            cpy_from_host(d_src[dev],src,bytes_device[dev], offset_device[dev],offset_host[dev], pixels_per_slice, buffer_length, stream[dev*nStream_device+1],  is_first_chunk,  is_last_chunk, image_size);
                        }
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);

                            stream[dev*nStream_device+1]->memcpy(d_u[dev],d_src[dev],mem_img_each_GPU);
                            stream[dev*nStream_device]->memset(d_px[dev],0,mem_img_each_GPU);
                            stream[dev*nStream_device]->memset(d_py[dev],0,mem_img_each_GPU);
                            stream[dev*nStream_device]->memset(d_pz[dev],0,mem_img_each_GPU);
                        }

                        // Sync
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            q->wait();
                        }
                    
                    }
                    // if we need to split and its not the first iteration, then we need to copy from Host memory.
                    // d_src is the original image, with no change.
                    if (splits>1 & i>0){

                        for (dev = 0; dev < deviceCount; dev++){ 
                            is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                            is_first_chunk=!(sp*deviceCount+dev);
                            q=&tigreSyclQueue(gpuids[dev]);
                            stream[dev*nStream_device+1]->wait();
                            if (is_last_chunk) {stream[dev*nStream_device+1]->memset(d_u[dev],0,mem_img_each_GPU);}
                            cpy_from_host(d_u[dev],h_u,bytes_device[dev], offset_device[dev],offset_host[dev], pixels_per_slice, buffer_length, stream[dev*nStream_device+1],  is_first_chunk,  is_last_chunk, image_size);
                        }

                        for (dev = 0; dev < deviceCount; dev++){ 
                            is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                            is_first_chunk=!(sp*deviceCount+dev);
                            q=&tigreSyclQueue(gpuids[dev]);
                            stream[dev*nStream_device+2]->wait();
                            if (is_last_chunk) {stream[dev*nStream_device+2]->memset(d_px[dev],0,mem_img_each_GPU);}
                            cpy_from_host(d_px[dev],h_px,bytes_device[dev], offset_device[dev],offset_host[dev], pixels_per_slice, buffer_length, stream[dev*nStream_device+2],  is_first_chunk,  is_last_chunk, image_size);
                        }
                        for (dev = 0; dev < deviceCount; dev++){ 
                            is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                            is_first_chunk=!(sp*deviceCount+dev);
                            q=&tigreSyclQueue(gpuids[dev]);
                            stream[dev*nStream_device+3]->wait();
                            if (is_last_chunk) {stream[dev*nStream_device+3]->memset(d_py[dev],0,mem_img_each_GPU);}
                            cpy_from_host(d_py[dev],h_py,bytes_device[dev], offset_device[dev],offset_host[dev], pixels_per_slice, buffer_length, stream[dev*nStream_device+3],  is_first_chunk,  is_last_chunk, image_size);
                        }
                        for (dev = 0; dev < deviceCount; dev++){ 
                            is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                            is_first_chunk=!(sp*deviceCount+dev);
                            q=&tigreSyclQueue(gpuids[dev]);
                            stream[dev*nStream_device+4]->wait();
                            if (is_last_chunk) {stream[dev*nStream_device+4]->memset(d_pz[dev],0,mem_img_each_GPU);}
                            cpy_from_host(d_pz[dev],h_pz,bytes_device[dev], offset_device[dev],offset_host[dev], pixels_per_slice, buffer_length, stream[dev*nStream_device+4],  is_first_chunk,  is_last_chunk, image_size);
                        }
                        // The lower halo of the first piece of this split: the slices the previous split had before it ran.
                        if (sp>0){
                            float** d_var[4]={d_u, d_px, d_py, d_pz};
                            for (int v = 0; v < 4; v++){
                                stream[1+v]->memcpy(d_var[v][0], carry+(size_t)v*buffer_pixels, buffer_pixels*sizeof(float));
                            }
                        }
                        for (dev = 0; dev < deviceCount; dev++){ 
                            is_last_chunk=!((sp*deviceCount+dev)<deviceCount*splits-1);
                            is_first_chunk=!(sp*deviceCount+dev);
                            q=&tigreSyclQueue(gpuids[dev]);
                            stream[dev*nStream_device+1]->wait();
                            if (is_last_chunk) {stream[dev*nStream_device+1]->memset(d_src[dev],0,mem_img_each_GPU);}
                            cpy_from_host(d_src[dev],src,bytes_device[dev], offset_device[dev],offset_host[dev], pixels_per_slice, buffer_length, stream[dev*nStream_device+1],  is_first_chunk,  is_last_chunk, image_size);
                        }

                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            q->wait();
                        }
                    }
                
                    // Before the last piece of this split runs, keep its top buffer_length slices for the next split.
                    // (Its interior starts at buffer_pixels and has slices_per_split >= buffer_length slices.)
                    if (splits>1 && sp<splits-1 && i>0){ // at i==0 every piece starts from src, no carry needed
                        float** d_var[4]={d_u, d_px, d_py, d_pz};
                        dev=deviceCount-1;
                        q=&tigreSyclQueue(gpuids[dev]);
                        for (int v = 0; v < 4; v++){
                            q->memcpy(carry+(size_t)v*buffer_pixels, d_var[v][dev]+(size_t)slices_per_split*pixels_per_slice, buffer_pixels*sizeof(float));
                        }
                        q->wait();
                    }

                    // Inter interations.
                    for(unsigned int ib=0;  (ib<(buffer_length)) && ((i+ib)<maxIter);  ib++){

                        tau2 = 0.3f + 0.02f * (i+ib);
                        tau1 = (1.f/tau2) * ((1.f/6.f) - (5.f/(15.f+(i+ib))));
                        // bdim and gdim
                    
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        
                            // Only the piece and its halo_lo/halo_hi halo slices (see above).
                            size_t off=(size_t)(buffer_length-halo_lo[dev])*pixels_per_slice;
                            long depth=(long)curr_slices+halo_lo[dev]+halo_hi[dev];
                            update_u(*stream[dev*nStream_device],d_src[dev]+off, d_pz[dev]+off, d_py[dev]+off, d_px[dev]+off, d_u[dev]+off, tau1, lambda,
                                    depth, image_size[1],image_size[0],
                                    spacing[2], spacing[1], spacing[0]);
                        }
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        
                            size_t off=(size_t)(buffer_length-halo_lo[dev])*pixels_per_slice;
                            long depth=(long)curr_slices+halo_lo[dev]+halo_hi[dev];
                            update_p(*stream[dev*nStream_device],d_u[dev]+off, d_pz[dev]+off, d_py[dev]+off, d_px[dev]+off, tau2,
                                    depth, image_size[1], image_size[0],
                                    spacing[2], spacing[1], spacing[0]);
                        }
                    }// END internal iter
                
                    // Synchronize mathematics, make sure bounding pixels are correct
                    for(dev=0; dev<deviceCount;dev++){
                        q=&tigreSyclQueue(gpuids[dev]);
                        q->wait();
                    }

                    // We have done as many iterations as our buffer allowed. We now need to syncronize the buffers.
            
                    if(splits==1){
                        // If everything fits in the GPUs, we can just share the updates between GPUs directly.
                        // We iterate for each device, and we copy the buffer pixels from each device. 
                        // "buffer" variables are just host auxiliary variables to allow the copy from GPU to GPU. 
                        // Essentially this code takes for each device (exceptions for the first and last devices included) "buffer_pixels" amount 
                        // of the beggining and end of each important variable and passes it to the next/previous GPU (the one containing the next/previous chunck)

                        // Pass buffer_pixels amount of data from the start of the image to the previous GPU
                        for(dev=0; dev<deviceCount;dev++){
                            if (dev<deviceCount-1){
                                q=&tigreSyclQueue(gpuids[dev+1]);
                                stream[(dev+1)*nStream_device+1]->memcpy(buffer_u,d_u[dev+1]+buffer_pixels,buffer_pixels*sizeof(float));
                                stream[(dev+1)*nStream_device+2]->memcpy(buffer_px,d_px[dev+1]+buffer_pixels,buffer_pixels*sizeof(float));
                                stream[(dev+1)*nStream_device+3]->memcpy(buffer_py,d_py[dev+1]+buffer_pixels,buffer_pixels*sizeof(float));
                                stream[(dev+1)*nStream_device+4]->memcpy(buffer_pz,d_pz[dev+1]+buffer_pixels,buffer_pixels*sizeof(float));

                            
                                q=&tigreSyclQueue(gpuids[dev]);
                                stream[(dev+1)*nStream_device+1]->wait();
                                stream[(dev)*nStream_device+1]->memcpy(d_u[dev] +slices_per_split*pixels_per_slice+buffer_pixels,buffer_u,buffer_pixels*sizeof(float));
                                stream[(dev+1)*nStream_device+2]->wait();
                                stream[(dev)*nStream_device+2]->memcpy(d_px[dev]+slices_per_split*pixels_per_slice+buffer_pixels,buffer_px,buffer_pixels*sizeof(float));
                                stream[(dev+1)*nStream_device+3]->wait();
                                stream[(dev)*nStream_device+3]->memcpy(d_py[dev]+slices_per_split*pixels_per_slice+buffer_pixels,buffer_py,buffer_pixels*sizeof(float));
                                stream[(dev+1)*nStream_device+4]->wait();
                                stream[(dev)*nStream_device+4]->memcpy(d_pz[dev]+slices_per_split*pixels_per_slice+buffer_pixels,buffer_pz,buffer_pixels*sizeof(float));
                            
                            
                            }
                            q->wait();
                            // Pass buffer_pixels amoung of data of the end part of the image to the next GPU.
                            if (dev>0){
                                // U
                                q=&tigreSyclQueue(gpuids[dev-1]);
                                stream[(dev-1)*nStream_device+1]->memcpy(buffer_u,d_u[dev-1] +slices_per_split*pixels_per_slice,buffer_pixels*sizeof(float));
                                stream[(dev-1)*nStream_device+2]->memcpy(buffer_px,d_px[dev-1]+slices_per_split*pixels_per_slice,buffer_pixels*sizeof(float));
                                stream[(dev-1)*nStream_device+3]->memcpy(buffer_py,d_py[dev-1]+slices_per_split*pixels_per_slice,buffer_pixels*sizeof(float));
                                stream[(dev-1)*nStream_device+4]->memcpy(buffer_pz,d_pz[dev-1]+slices_per_split*pixels_per_slice,buffer_pixels*sizeof(float));
                            
                            
                                q=&tigreSyclQueue(gpuids[dev]);
                                stream[(dev-1)*nStream_device+1]->wait();
                                stream[(dev)*nStream_device+1]->memcpy(d_u[dev],buffer_u,buffer_pixels*sizeof(float));
                                stream[(dev-1)*nStream_device+2]->wait();
                                stream[(dev)*nStream_device+2]->memcpy(d_px[dev],buffer_px,buffer_pixels*sizeof(float));
                                stream[(dev-1)*nStream_device+3]->wait();
                                stream[(dev)*nStream_device+3]->memcpy(d_py[dev],buffer_py,buffer_pixels*sizeof(float));
                                stream[(dev-1)*nStream_device+4]->wait();
                                stream[(dev)*nStream_device+4]->memcpy(d_pz[dev],buffer_pz,buffer_pixels*sizeof(float));
                                q->wait();
                            
                            
                            }
                        }
                    // This is the case when we can't solely use GPU memory, as the total size of the images+variables exceeds total amounf of memory among GPUs.
                    // This situation requires partial results and full memory allocation in the host. 
                    }else{
                        // Vopy all the U variable into the host.
                        for(dev=0; dev<deviceCount;dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices      = ((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            linear_idx_start = pixels_per_slice*slices_per_split*(sp*deviceCount+dev);
                            total_pixels     = curr_slices*pixels_per_slice;
                            stream[dev*nStream_device+1]->memcpy(&h_u[linear_idx_start],d_u [dev]+buffer_pixels,total_pixels*sizeof(float));
                        }
                        if ((i+buffer_length)<maxIter){ // If its the last iteration, we don't need to get these out.
                            // if its not, copy them to host fully. 
                            for(dev=0; dev<deviceCount;dev++){
                                q=&tigreSyclQueue(gpuids[dev]);
                                curr_slices      = ((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                                linear_idx_start = pixels_per_slice*slices_per_split*(sp*deviceCount+dev);
                                total_pixels     = curr_slices*pixels_per_slice;
                                stream[dev*nStream_device+2]->memcpy(&h_px[linear_idx_start],d_px[dev]+buffer_pixels,total_pixels*sizeof(float));
                                stream[dev*nStream_device+3]->memcpy(&h_py[linear_idx_start],d_py[dev]+buffer_pixels,total_pixels*sizeof(float));
                                stream[dev*nStream_device+4]->memcpy(&h_pz[linear_idx_start],d_pz[dev]+buffer_pixels,total_pixels*sizeof(float));
                            
                            }
                        }
                        // Wait for all devices to finish copying to the host before the next split reads from it:
                        // the first device of the next split reads its halo from what the last device of this split is writing.
                        for(dev=0; dev<deviceCount;dev++){
                            tigreSyclQueue(gpuids[dev]).wait();
                        }

                    }
                }//END splits
            }//END main iter
        
            for(dev=0; dev<deviceCount;dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                q->wait();
            }
        
            // We are done. If we were solely using GPU memory, because the problem fitted fully on all GPU memory available, then the result is still inside the GPU.
            // lets get it out. 
            if(splits==1){
                for(dev=0; dev<deviceCount;dev++){
                    q=&tigreSyclQueue(gpuids[dev]);
                    curr_slices  = ((dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*dev;
                    total_pixels = curr_slices*pixels_per_slice;
                    stream[dev*nStream_device+1]->memcpy(dst+slices_per_split*pixels_per_slice*dev,d_u[dev]+buffer_pixels,total_pixels*sizeof(float));
                }
            } // done, everything in GPU and auxiliary variables are good to go. 

            for(dev=0; dev<deviceCount;dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                q->wait();
            }
            onError.dismiss();  // from here on the function releases everything itself
            for(dev=0; dev<deviceCount;dev++){
            
                sycl::free(d_src[dev],tigreSyclQueue(gpuids[dev]));
                sycl::free(d_u [dev],tigreSyclQueue(gpuids[dev]));
                sycl::free(d_pz[dev],tigreSyclQueue(gpuids[dev]));
                sycl::free(d_py[dev],tigreSyclQueue(gpuids[dev]));
                sycl::free(d_px[dev],tigreSyclQueue(gpuids[dev]));
            }
            if(splits>1 && buffer_length<maxIter){
                free(h_px);
                free(h_py);
                free(h_pz);
                free(carry);
            }else if(splits==1){
                free(buffer_u);
                free(buffer_px);
                free(buffer_py);
                free(buffer_pz);
            }
        
            free(stream);

            for(dev=0; dev<deviceCount;dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                q->wait();
            }
        
    
    } catch (const sycl::exception& e) {
        mexErrMsgIdAndTxt("tvDenoise:tvdenoising:GPU", e.what());
    }
}
    

void checkFreeMemory(const GpuIds& gpuids,size_t *mem_GPU_global){
        int deviceCount = gpuids.GetLength();
        for (int dev = 0; dev < deviceCount; dev++){
            size_t memfree=tigreSyclCheckFreeMemory(gpuids[dev],"tvDenoise:tvdenoising:GPU");
            if(dev==0) *mem_GPU_global=memfree;
            *mem_GPU_global=(memfree<*mem_GPU_global)?memfree:*mem_GPU_global;
        }
}

void cpy_from_host(float* device_array,float* host_array, 
                   unsigned long long bytes_device,unsigned long long offset_device,unsigned long long offset_host, 
                   unsigned long long pixels_per_slice, unsigned int buffer_length, 
                   sycl::queue* stream, bool is_first_chunk, bool is_last_chunk,const long* image_size)
{

    // The halo slots beyond the image's first and last slice are not used (the kernels skip them and apply
    // their own boundary conditions there), so nothing is copied into them.
    stream->wait();
    stream->memcpy(device_array +offset_device,host_array +offset_host,bytes_device*sizeof(float));
}
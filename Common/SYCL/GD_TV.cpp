/*-------------------------------------------------------------------------
 *
 * SYCL functions for Steepest descend in POCS-type algorithms.
 *
 * This file will iteratively minimize by steepest descend the total variation
 * of the input image, with the parameters given, using GPUs.
 *
 * CODE by       Ander Biguri
 *
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







#define MAXTHREADS 1024
#define MAX_BUFFER 60

#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include "GD_TV.hpp"
#include "tv_gradient_descent.hpp"
#include "gpuUtils.hpp"
#include "tigre_sycl.hpp"
#include "tigre_release_on_error.hpp"

// The kernels of each TIGRE module share one SYCL runtime in a Python process, and the
// runtime finds kernels by name. A named namespace keeps these kernel names distinct
// from the identically named helpers of the other TV modules.
namespace tigre_gd_tv {
// SYCL kernels
    // Backward differences at voxel (z,y,x) of the buffer; 0 outside the image. In x and y the image is
    // [0,cols) and [0,rows); in z it is the slots [zlo,zhi) of the buffer (see TVPiece in
    // tv_gradient_descent.hpp), as in Common/CUDA/GD_TV.cu.
    static inline void gradient(const float* u, float* grad,
            long z, long y, long x,
            long zlo, long zhi, long rows, long cols){
        if (z < zlo || z >= zhi || y >= rows || x >= cols)
            return;
        unsigned long size2d = rows*cols;
        size_t idx = (size_t)z * size2d + y * cols + x;
        
        float uidx = u[idx];
        
        if ( z - 1 >= zlo) {
            grad[0] = (uidx-u[(size_t)(z-1)*size2d + y*cols + x]) ;
        }
        
        if ( y - 1 >= 0){
            grad[1] = (uidx-u[(size_t)z*size2d + (y-1)*cols + x]) ;
        }
        
        if ( x - 1 >= 0) {
            grad[2] = (uidx-u[(size_t)z*size2d + y*cols + (x-1)]);
        }
    }
    
    static inline void gradientTV(const float* f, float* dftv,
            long depth, long zlo, long zhi, long rows, long cols,
            unsigned long x, unsigned long y, unsigned long z){
        size_t idx = (size_t)z * (rows * cols) + y * cols + x;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        
        
        float df[3] ={0.f,0.f,0.f};
        float dfi[3]={0.f,0.f,0.f}; // dfi== \partial f_{i+1,j,k}
        float dfj[3]={0.f,0.f,0.f};
        float dfk[3]={0.f,0.f,0.f};
        gradient(f,df  ,z  ,y  ,x  , zlo,zhi,rows,cols);
        gradient(f,dfi ,z  ,y  ,x+1, zlo,zhi,rows,cols);
        gradient(f,dfj ,z  ,y+1,x  , zlo,zhi,rows,cols);
        gradient(f,dfk ,z+1,y  ,x  , zlo,zhi,rows,cols);
        float eps=0.00000001; //% avoid division by zero
        
        dftv[idx]=(df[0]+df[1]+df[2])/(sycl::sqrt(df[0] *df[0] +df[1] *df[1] +df[2] *df[2])+eps)
        -dfi[2]/(sycl::sqrt(dfi[0]*dfi[0]+dfi[1]*dfi[1]+dfi[2]*dfi[2]) +eps)     // I wish I coudl precompute this, but if I do then Id need to recompute the gradient.
        -dfj[1]/(sycl::sqrt(dfj[0]*dfj[0]+dfj[1]*dfj[1]+dfj[2]*dfj[2]) +eps)
        -dfk[0]/(sycl::sqrt(dfk[0]*dfk[0]+dfk[1]*dfk[1]+dfk[2]*dfk[2]) +eps);
        return;
        
    }
    

    // Kernel launches. The CUDA grid/block shapes and the two-pass reduction
    // map onto SYCL ranges and sycl::reduction.
    static void gradientTV(sycl::queue& q, const float* f, float* dftv,
            long depth, long zlo, long zhi, long rows, long cols){
        q.parallel_for(sycl::range<3>(depth, rows, cols), [=](sycl::id<3> id){
            gradientTV(f,dftv,depth,zlo,zhi,rows,cols,id[2],id[1],id[0]);
        });
    }
    static void divideArrayScalar(sycl::queue& q, float* vec, float scalar, const size_t n){
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ vec[i]/=scalar; });
    }
    static void multiplyArrayScalar(sycl::queue& q, float* vec, float scalar, const size_t n){
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ vec[i]*=scalar; });
    }
    static void substractArrays(sycl::queue& q, float* vec, float* vec2, const size_t n){
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ vec[i]-=vec2[i]; });
    }
    // Sum of squares of g_idata[0..n) into g_odata[0].
    // Sum of squares of g_idata (nslices slices of slice_px values), one float per block of
    // MAXTHREADS values, the blocks restarting at each slice, into g_odata[0..nslices*blocks per slice).
    // The host adds the blocks in double (#856). As the blocks follow the slices, each slice gives the
    // same float sums however the image is shared between devices.
    static void reduceNorm2(sycl::queue& q, const float* g_idata, float* g_odata, size_t nslices, size_t slice_px){
        size_t bps=(slice_px+MAXTHREADS-1)/MAXTHREADS;
        size_t wg=q.get_device().get_info<sycl::info::device::max_work_group_size>();
        wg=wg<256? wg:256;
        while (MAXTHREADS%wg) wg--;
        q.parallel_for(sycl::nd_range<1>(nslices*bps*wg, wg), [=](sycl::nd_item<1> it){
            size_t g=it.get_group(0);
            size_t slice=g/bps;
            size_t blk=g%bps;
            size_t lid=it.get_local_id(0);
            float s=0;
            for (size_t k=lid; k<MAXTHREADS; k+=wg){
                size_t in_slice=blk*MAXTHREADS+k;
                if (in_slice<slice_px){ float v=g_idata[slice*slice_px+in_slice]; s+=v*v; }
            }
            s=sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            if (lid==0) g_odata[g]=s;
        });
    }

}  // namespace tigre_gd_tv
using namespace tigre_gd_tv;

// main function
    // See tv_gradient_descent.hpp. (This was minTV's own function; minTV, AwminTV and PICCS now share it.)
    void tv_gradient_descent(const float* img, const float* const* extra, int nExtra,
                             TVStepDirection step, const void* params,
                             float* dst, float alpha, const long* image_size, int maxIter, const GpuIds& gpuids,
                             const char* what){
    try {
        
        
       
        
            // Prepare for MultiGPU
            int deviceCount = gpuids.GetLength();
            if (deviceCount == 0) {
                mexErrMsgIdAndTxt("GD_TV:GPU","There are no available device(s) that support SYCL\n");
            }
            //
            // CODE assumes
            // 1.-All available devices are usable by this code
            // 2.-All available devices are equal, they are the same machine (warning thrown)
            // Check the available devices, and if they are the same
            if (!gpuids.AreEqualDevices()) {
                mexWarnMsgIdAndTxt("minimizeTV:GD_TV:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
            }
        
            int dev;
            sycl::queue* q=&tigreSyclQueue(gpuids[0]);
        
            // We don't know if the devices are being used. lets check that. and only use the amount of memory we need.

            size_t mem_GPU_global;
            checkFreeMemory(gpuids, &mem_GPU_global);

        
        
            // %5 of free memory should be enough, we have almost no variables in these kernels
            size_t total_pixels              = (size_t)image_size[0]*image_size[1]  * image_size[2] ;
            size_t mem_slice_image           = sizeof(float)* (size_t)image_size[0]*image_size[1]  ;
            size_t mem_size_image            = sizeof(float)* total_pixels;
            // Image-sized buffers on each GPU: image, step direction, its copy for the norm, and the extra inputs.
            const unsigned int nImages = 3 + (unsigned int)nExtra;
            size_t mem_auxiliary             = sizeof(float)* (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
        
            // Decide how are we handling the distribution of computation
            size_t mem_img_each_GPU;
        
            unsigned int buffer_length=2;
            //Does everything fit in the GPU?
            unsigned int slices_per_split;
        
            // if it is a thin problem (no need to split), just use one GPU
            if (image_size[2]<4){deviceCount=1;}

            unsigned int splits=1; // if the number does not fit in an uint, you have more serious trouble than this.
            if(mem_GPU_global> nImages*mem_size_image+nImages*(deviceCount-1)*mem_slice_image*buffer_length+mem_auxiliary){
                // We only need to split if we have extra GPUs
                slices_per_split=(image_size[2]+deviceCount-1)/deviceCount;
                // No empty pieces (see tvPiecesWithoutEmpty); this may use fewer GPUs or smaller pieces.
                tvPiecesWithoutEmpty((unsigned long)image_size[2], &deviceCount, &splits, &slices_per_split);
                mem_img_each_GPU=mem_slice_image*((slices_per_split+buffer_length*2));
            }else{
                // The smallest working set: on each GPU, 3 images of one slice plus the
                // minimum overlap (2 slices on each side), and the auxiliary memory. If
                // not even that fits, say so now: the sizes below are unsigned and would
                // wrap around past zero.
                const size_t mem_min=nImages*mem_slice_image*(1+2*2)+mem_auxiliary;
                if (mem_GPU_global<=mem_min)
                    tigreNotEnoughGpuMemory("GD_TV:GPUmemory",what,mem_min,mem_GPU_global);
                // As mem_auxiliary is not expected to be a large value (for a 2000^3 image is around 28Mbytes), lets for now assume we need it all
                size_t mem_free=mem_GPU_global-mem_auxiliary;
            
                splits=(unsigned int)(ceil(((float)(nImages*mem_size_image)/(float)(deviceCount))/mem_free));
                // Now, there is an overhead here, as each splits should have 2 slices more, to account for overlap of images.
                // lets make sure these 2 slices fit, if they do not, add 1 to splits.
                slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits);
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
            
                // if the new stuff does not fit in the GPU, it means we are in the edge case where adding that extra slice will overflow memory
                // One more split usually does it; with very little memory it may take more
                // (one slice per piece always fits, checked above).
                while (mem_GPU_global< nImages*mem_img_each_GPU+mem_auxiliary && slices_per_split>1){
                    splits++;
                    //recompute for later
                    slices_per_split=(image_size[2]+deviceCount*splits-1)/(deviceCount*splits); // amount of slices that fit on a GPU. Later we add 2 to these, as we need them for overlap
                    mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));
                }
                // No empty pieces (see tvPiecesWithoutEmpty); this may use fewer GPUs or smaller pieces.
                tvPiecesWithoutEmpty((unsigned long)image_size[2], &deviceCount, &splits, &slices_per_split);
                mem_img_each_GPU=(mem_slice_image*(slices_per_split+buffer_length*2));


                // How many EXTRA buffer slices should be able to fit in here??!?!
                // Only do it if there are splits needed. 
                if(splits>1){
                    mem_free=mem_GPU_global-(nImages*mem_img_each_GPU+mem_auxiliary);
                    unsigned int extra_buff=(mem_free/mem_slice_image); 
                    buffer_length=(extra_buff/2)/nImages; // we need double whatever this results in, rounded down.
                    buffer_length=std::max(buffer_length,2u);// minimum 2
                    buffer_length=std::min((unsigned int)MAX_BUFFER,buffer_length);

                    mem_img_each_GPU=mem_slice_image*(slices_per_split+buffer_length*2);
                
                }else{
                    buffer_length=2;
                }

                // Assert
                if (mem_GPU_global< nImages*mem_img_each_GPU+mem_auxiliary){
                    mexErrMsgIdAndTxt("GD_TV:GPU","Assertion Failed. Logic behind splitting flawed! Please tell: ander.biguri@gmail.com\n");
                }
            }
        
        
             // Assert
       
            if ((slices_per_split+buffer_length*2)*(size_t)image_size[0]*image_size[1]* sizeof(float)!= mem_img_each_GPU){
                mexErrMsgIdAndTxt("GD_TV:GPU","Assertion Failed. Memory needed calculation broken! Please tell: ander.biguri@gmail.com\n");
            }
        
        
        
        
        
        
            // onError releases what the call holds if it is left by an error (see
            // tigre_release_on_error.hpp), as in the CUDA build (#870).
            tigre::ReleaseOnError onError;
            for (int d = 0; d < deviceCount; d++) onError.addQueue(&tigreSyclQueue(gpuids[d]));
            float** d_image=    (float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_image]{ free(p); });
            float** d_dimgTV=   (float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_dimgTV]{ free(p); });
            float** d_norm2aux= (float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_norm2aux]{ free(p); });
            float** d_norm2=    (float**)malloc(deviceCount*sizeof(float*));
            onError.add([p=d_norm2]{ free(p); });
            // The extra input volumes on each GPU (d_extra[dev*nExtra+k]), split and haloed like the image.
            float** d_extra=    (float**)calloc(deviceCount*nExtra+1,sizeof(float*));
            onError.add([p=d_extra]{ free(p); });
         
            // Room for the norm's block sums of one piece (see reduceNorm2).
            const size_t blocks_per_slice=((size_t)image_size[0]*image_size[1]+MAXTHREADS-1)/MAXTHREADS;
            const size_t mem_norm_parts=std::max(mem_auxiliary, sizeof(float)*blocks_per_slice*slices_per_split);
            // allocate memory in each GPU
            for (dev = 0; dev < deviceCount; dev++){
                q=&tigreSyclQueue(gpuids[dev]);
            
                d_image[dev]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "GD_TV:GPU", what);
                onError.add([p=d_image[dev], q]{ sycl::free(p, *q); });
                q->memset(d_image[dev],0,mem_img_each_GPU).wait();
                d_dimgTV[dev]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "GD_TV:GPU", what);
                onError.add([p=d_dimgTV[dev], q]{ sycl::free(p, *q); });
                q->memset(d_dimgTV[dev],0,mem_img_each_GPU).wait();
                d_norm2[dev]=(float*)tigre::deviceMallocBytes(slices_per_split*mem_slice_image, gpuids[dev], "GD_TV:GPU", what);
                onError.add([p=d_norm2[dev], q]{ sycl::free(p, *q); });
                q->memset(d_norm2[dev],0,slices_per_split*mem_slice_image).wait();
                d_norm2aux[dev]=(float*)tigre::deviceMallocBytes(mem_norm_parts, gpuids[dev], "GD_TV:GPU", what);
                onError.add([p=d_norm2aux[dev], q]{ sycl::free(p, *q); });
                q->memset(d_norm2aux[dev],0,mem_norm_parts).wait();
                for (int k = 0; k < nExtra; k++){
                    d_extra[dev*nExtra+k]=(float*)tigre::deviceMallocBytes(mem_img_each_GPU, gpuids[dev], "GD_TV:GPU", what);
                    onError.add([p=d_extra[dev*nExtra+k], q]{ sycl::free(p, *q); });
                    q->memset(d_extra[dev*nExtra+k],0,mem_img_each_GPU).wait();
                }
            
            
            }
           unsigned long long buffer_pixels=buffer_length*(size_t)image_size[0]*image_size[1];
            float* buffer;
            if(splits>1){
                mexWarnMsgIdAndTxt("minimizeTV:GD_TV:Image_split","Your image does not fit on the available GPUs. The TV minimization will be significantly slowed due to the image size.\nApproximated mathematics turned on for computational speed.");
            }else{
                buffer=(float*)malloc(buffer_length*(size_t)image_size[0]*image_size[1]*sizeof(float));
                onError.add([p=buffer]{ free(p); });
            }
        
        
        

        
        
                    // Create streams
            int nStream_device=2;
            int nStreams=deviceCount*nStream_device;
            sycl::queue** stream=(sycl::queue**)malloc(nStreams*sizeof(sycl::queue*));
            onError.add([p=stream]{ free(p); });
        
            for (dev = 0; dev < deviceCount; dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                for (int i = 0; i < nStream_device; ++i){
                    stream[i+dev*nStream_device]=q;
                }
            }

        
            // For the reduction

            double totalsum_prev;
            double totalsum;
            // Each device's part of the norm and the sum of the parts are kept in double (#856),
            // so the norm barely depends on how the image is split between devices.
            double sum_curr_spl;
            double * sumnorm2;
            sumnorm2=(double*)malloc(deviceCount*sizeof(double));
            onError.add([p=sumnorm2]{ free(p); });
            size_t nparts_max=mem_norm_parts/sizeof(float);
            float * normparts;
            normparts=(float*)malloc(deviceCount*nparts_max*sizeof(float));
            onError.add([p=normparts]{ free(p); });
        
            unsigned int curr_slices;
            unsigned long long curr_pixels;
            size_t linear_idx_start;
            unsigned long long* offset_device=(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
            onError.add([p=offset_device]{ free(p); });
            unsigned long long* offset_host  =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
            onError.add([p=offset_host]{ free(p); });
            unsigned long long* bytes_device =(unsigned long long*)malloc(deviceCount*sizeof(unsigned long long));
            onError.add([p=bytes_device]{ free(p); });
            // Halo slices below and above each piece: buffer_length, or fewer where the piece is closer than that to the
            // image's first or last slice, as in tv_proximal.cpp and Common/CUDA/GD_TV.cu (#899). A piece's interior
            // always starts at buffer_pixels on the device; the halo_lo[dev] slices below it and halo_hi[dev] above it
            // hold the image.
            unsigned int* halo_lo=(unsigned int*)malloc(deviceCount*sizeof(unsigned int));
            onError.add([p=halo_lo]{ free(p); });
            unsigned int* halo_hi=(unsigned int*)malloc(deviceCount*sizeof(unsigned int));
            onError.add([p=halo_hi]{ free(p); });
            const size_t slice_pixels=(size_t)image_size[0]*image_size[1];
            for(unsigned int i=0;i<maxIter;i+=(buffer_length-1)){
                if(splits>1){
                    totalsum_prev=0;
                }
                for(unsigned int sp=0;sp<splits;sp++){
                
                    // For each iteration we need to compute all the image. The ordering of these loops
                    // need to be like this due to the bounding layers between splits. If more than 1 split is needed
                    // for each GPU then there is no other way that taking the entire memory out of GPU and putting it back.
                    // If the memory can be shared between GPUs fully without extra splits, then there is an easy way of synchronizing the memory
                
                    // Copy image to memory
                    for (dev = 0; dev < deviceCount; dev++){
                        curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                        curr_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                        linear_idx_start=(size_t)image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                    
                        // lets compute where we start copyes and how much. This avoids 3 calls to Memcpy
                        size_t start_slice=(size_t)slices_per_split*(sp*deviceCount+dev);
                        halo_lo[dev]=(unsigned int)std::min((size_t)buffer_length, start_slice);
                        halo_hi[dev]=(unsigned int)std::min((size_t)buffer_length, (size_t)image_size[2]-start_slice-curr_slices);
                        offset_device[dev]=(buffer_length-halo_lo[dev])*slice_pixels;
                        offset_host[dev]=linear_idx_start-halo_lo[dev]*slice_pixels;
                        bytes_device[dev]=curr_pixels+(halo_lo[dev]+halo_hi[dev])*slice_pixels;
                    }

                    if(i==0){
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                        
                            stream[dev*nStream_device+1]->memcpy(d_image[dev]+offset_device[dev],img+offset_host[dev],bytes_device[dev]*sizeof(float));
                            for (int k = 0; k < nExtra; k++)
                                stream[dev*nStream_device+1]->memcpy(d_extra[dev*nExtra+k]+offset_device[dev],extra[k]+offset_host[dev],bytes_device[dev]*sizeof(float));
                        
                        
                        }
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            q->wait();
                        }
                    }
                    // if we need to split and its not the first iteration, then we need to copy from Host memory the previosu result.
                    if (splits>1 & i>0){
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            stream[dev*nStream_device+1]->memcpy(d_image[dev]+offset_device[dev],dst+offset_host[dev],bytes_device[dev]*sizeof(float));
                            for (int k = 0; k < nExtra; k++)  // each piece has its own part of the extra inputs
                                stream[dev*nStream_device+1]->memcpy(d_extra[dev*nExtra+k]+offset_device[dev],extra[k]+offset_host[dev],bytes_device[dev]*sizeof(float));
                        
                        
                        }
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            q->wait();
                        }
                    }
                
                    for(unsigned int ib=0;  (ib<(buffer_length-1)) && ((i+ib)<maxIter);  ib++){
                    
                        // For the gradient
                    
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            // Compute the gradient of the TV norm
                        
                            // The step direction for this piece (the caller's functional). The image's slices in the buffer
                            // are [zlo,zhi): the halo slots that would lie before the image's first slice or after its last
                            // are outside it (see halo_lo, halo_hi).
                            TVPiece pc;
                            pc.image=d_image[dev];
                            pc.extra=d_extra+dev*nExtra;
                            pc.direction=d_dimgTV[dev];
                            pc.depth=(long)(curr_slices+buffer_length*2-1);
                            pc.zlo=(long)(buffer_length-halo_lo[dev]);
                            pc.zhi=(long)(buffer_length+curr_slices+halo_hi[dev]);
                            if (pc.zhi>pc.depth) pc.zhi=pc.depth;
                            pc.rows=image_size[1];
                            pc.cols=image_size[0];
                            pc.interior_offset=buffer_pixels;
                            pc.interior_pixels=(size_t)curr_slices*image_size[0]*image_size[1];
                            pc.stream=stream[dev*nStream_device];
                            step(pc, params);
                        
                        }
                    
                    
                    
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            // no need to copy the 2 aux slices here
                            stream[dev*nStream_device]->wait();
                            stream[dev*nStream_device+1]->memcpy(d_norm2[dev],d_dimgTV[dev]+buffer_pixels,(size_t)image_size[0]*image_size[1]*curr_slices*sizeof(float));
                        }
                    
                    
                        // Compute the L2 norm of the gradient. For that, reduction is used.
                        //REDUCE
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                        
                        
                            stream[dev*nStream_device+1]->wait();
                            reduceNorm2(*stream[dev*nStream_device],d_norm2[dev], d_norm2aux[dev], curr_slices, (size_t)image_size[0]*image_size[1]);
                        
                        }
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];

                            stream[dev*nStream_device]->wait();
                            stream[dev*nStream_device+1]->memcpy(normparts+dev*nparts_max, d_norm2aux[dev], curr_slices*blocks_per_slice*sizeof(float));
                        }
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            q->wait();
                         }
                        for (dev = 0; dev < deviceCount; dev++){
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                            sumnorm2[dev]=0;
                            for (size_t k=0; k<curr_slices*blocks_per_slice; k++){
                                sumnorm2[dev]+=normparts[dev*nparts_max+k];
                            }
                        }
                    
                    
                        // Accumulate the norm accross devices
                        sum_curr_spl=0;
                        // this is CPU code
                        for (dev = 0; dev < deviceCount; dev++){
                            sum_curr_spl+=sumnorm2[dev];
                        }
                        sum_curr_spl+=0.0000001; // avoid division by zero
                    
                        // If we have more than one splits, lets use the result from prior calls
                        if(i>0 && splits>1){
                            // this is already stored:
                            //totalsum=totalsum_prev; 
                        }else{
                            totalsum=sum_curr_spl;
                        }
                    
                    
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                            //NORMALIZE
                            //in a Tesla, maximum blocks =15 SM * 4 blocks/SM
                            divideArrayScalar(*stream[dev*nStream_device],d_dimgTV[dev]+buffer_pixels,(float)sqrt(totalsum),total_pixels);
                            //MULTIPLY HYPERPARAMETER
                            multiplyArrayScalar(*stream[dev*nStream_device],d_dimgTV[dev]+buffer_pixels,alpha,   total_pixels);
                        }
                         for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            q->wait();
                         }
                    
                        //SUBSTRACT GRADIENT
                        //////////////////////////////////////////////
                        for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                        
                            substractArrays(*stream[dev*nStream_device],d_image[dev]+buffer_pixels,d_dimgTV[dev]+buffer_pixels, total_pixels);
                        }
                    }

                    // Synchronize mathematics, make sure bounding pixels are correct
                     for (dev = 0; dev < deviceCount; dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                            q->wait();
                         }
                
                    if(splits==1){
                        for(dev=0; dev<deviceCount;dev++){
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                            if (dev<deviceCount-1){
                                q=&tigreSyclQueue(gpuids[dev+1]);
                                q->memcpy(buffer,d_image[dev+1]+buffer_pixels,buffer_pixels*sizeof(float)).wait();
                                q=&tigreSyclQueue(gpuids[dev]);
                                q->memcpy(d_image[dev]+total_pixels+buffer_pixels,buffer,buffer_pixels*sizeof(float)).wait(); 
                            }
                            q->wait();
                            if (dev>0){
                                q=&tigreSyclQueue(gpuids[dev-1]);
                                q->memcpy(buffer,d_image[dev-1]+(size_t)slices_per_split*(size_t)image_size[0]*image_size[1],buffer_pixels*sizeof(float));
                                q->wait();
                                q=&tigreSyclQueue(gpuids[dev]);
                                q->memcpy(d_image[dev],buffer,buffer_pixels*sizeof(float));
                                q->wait();
                            }
                        }
                    }else{
                    
                        // We need to take it out :(
                        for(dev=0; dev<deviceCount;dev++){
                            q=&tigreSyclQueue(gpuids[dev]);
                        
                            curr_slices=((sp*deviceCount+dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*(sp*deviceCount+dev);
                            linear_idx_start=(size_t)image_size[0]*image_size[1]*slices_per_split*(sp*deviceCount+dev);
                            total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                            stream[dev*nStream_device+1]->memcpy(&dst[linear_idx_start],d_image[dev]+buffer_pixels,total_pixels*sizeof(float));
                        }
                    }
                
                    for (dev = 0; dev < deviceCount; dev++){
                        q=&tigreSyclQueue(gpuids[dev]);
                        q->wait();
                    }

                    totalsum_prev+=sum_curr_spl;
                }
                totalsum=totalsum_prev;
            }
            // If there has not been splits, we still have data in memory
            if(splits==1){
                for(dev=0; dev<deviceCount;dev++){
                    q=&tigreSyclQueue(gpuids[dev]);
                
                    curr_slices=((dev+1)*slices_per_split<image_size[2])?  slices_per_split:  image_size[2]-slices_per_split*dev;
                    total_pixels=curr_slices*(size_t)image_size[0]*image_size[1];
                    q->memcpy(dst+slices_per_split*(size_t)image_size[0]*image_size[1]*dev,d_image[dev]+buffer_pixels,total_pixels*sizeof(float)).wait();
                }
            }
        
            onError.dismiss();  // from here on the function releases everything itself
            for(dev=0; dev<deviceCount;dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                sycl::free(d_image[dev],tigreSyclQueue(gpuids[dev]));
                sycl::free(d_norm2aux[dev],tigreSyclQueue(gpuids[dev]));
                sycl::free(d_dimgTV[dev],tigreSyclQueue(gpuids[dev]));
                sycl::free(d_norm2[dev],tigreSyclQueue(gpuids[dev]));
                for (int k = 0; k < nExtra; k++) sycl::free(d_extra[dev*nExtra+k],tigreSyclQueue(gpuids[dev]));
            }
            if (splits==1){
                free(buffer);
            }
            // these were freed only on error
            free(d_image); free(d_dimgTV); free(d_norm2aux); free(d_norm2); free(d_extra);
            free(offset_device); free(offset_host); free(bytes_device); free(halo_lo); free(halo_hi);
            free(sumnorm2); free(normparts);
        
            free(stream);
        
            for (dev = 0; dev < deviceCount; dev++){
                q=&tigreSyclQueue(gpuids[dev]);
                q->wait();
            }
    
    } catch (const sycl::exception& e) {
        mexErrMsgIdAndTxt("GD_TV:GPU", e.what());
    }
}

// minTV: the step direction is the gradient of the TV norm.
static void tvStepDirection(const TVPiece& pc, const void*){
    gradientTV(*pc.stream, pc.image, pc.direction, pc.depth, pc.zlo, pc.zhi, pc.rows, pc.cols);
}

void pocs_tv(float* img,float* dst,float alpha,const long* image_size, int maxIter, const GpuIds& gpuids){
    tv_gradient_descent(img, NULL, 0, tvStepDirection, NULL, dst, alpha, image_size, maxIter, gpuids, "Total variation minimization (minTV)");
}
        
void checkFreeMemory(const GpuIds& gpuids,size_t *mem_GPU_global){
        int deviceCount = gpuids.GetLength();
        for (int dev = 0; dev < deviceCount; dev++){
            size_t memfree=tigreSyclCheckFreeMemory(gpuids[dev],"GD_TV:GPU");
            if(dev==0) *mem_GPU_global=memfree;
            *mem_GPU_global=(memfree<*mem_GPU_global)?memfree:*mem_GPU_global;
        }
}

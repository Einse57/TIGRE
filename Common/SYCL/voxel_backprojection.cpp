/*-------------------------------------------------------------------------
 *
 * SYCL function for backrpojection using FDK weigts for CBCT
 *
 *
 * CODE by  Ander Biguri
 *          Optimized and modified by RB
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

#define  PI_2 1.57079632679489661923
#include <algorithm>
#include <vector>
#include <sycl/sycl.hpp>
#include "voxel_backprojection.hpp"
#include "TIGRE_common.hpp"
#include "tigre_sycl.hpp"
#include "tigre_release_on_error.hpp"
#include "page_locking.hpp"
#include <math.h>
#include "GpuIds.hpp"

#define MAXTREADS 1024
    /*GEOMETRY DEFINITION
     *
     *                Detector plane, behind
     *            |-----------------------------|
     *            |                             |
     *            |                             |
     *            |                             |
     *            |                             |
     *            |      +--------+             |
     *            |     /        /|             |
     *   A Z      |    /        / |*D           |
     *   |        |   +--------+  |             |
     *   |        |   |        |  |             |
     *   |        |   |     *O |  +             |
     *   *--->y   |   |        | /              |
     *  /         |   |        |/               |
     * V X        |   +--------+                |
     *            |-----------------------------|
     *
     *           *S
     *
     *
     *
     *
     *
     **/
    


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RB, 10/31/2016: Add constant memory arrays to store parameters for all projections to be analyzed during a single kernel call
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The optimal values of two constants obtained by RB on NVIDIA Quadro K2200 (4 GB RAM, 640 CUDA cores) for 512^3 volume and 512^3 projections (512 proj, each 512 x 512) were:
// PROJ_PER_KERNEL = 32 or 16 (very similar times)
// VOXELS_PER_THREAD = 8
// Speedup of the entire FDK backprojection (not only kernel run, also memcpy etc.) was nearly 4x relative to the original (single projection, single voxel per thread) code.
// (e.g. 16.2 s vs. ~62 s).

const int PROJ_PER_KERNEL = 32;  // Number of 2D projections to be analyzed by a single thread. This can be tweaked to see what works best. 32 was the optimal value in the paper by Zinsser and Keck.
const int VOXELS_PER_THREAD = 8;  // Number of voxels to be computed by s single thread. Can be tweaked to see what works best. 4 was the optimal value in the paper by Zinsser and Keck.

// We have PROJ_PER_KERNEL projections and we need 6 parameters for each projection:
//   deltaX, deltaY, deltaZ, xyzOrigin, offOrig, offDetec
// So we need to keep PROJ_PER_KERNEL*6 values in our deltas array FOR EACH CALL to our main kernel
// (they will be updated in the main loop before each kernel call).


// We also need a corresponding array on the host side to be filled before each kernel call, then copied to the device (array in constant memory above)
// Point3D projParamsArrayHost[6*PROJ_PER_KERNEL];   // Host means it is host memory

// Now we also need to store sinAlpha and cosAlpha for each projection (two floats per projection)



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// END RB, 10/31/2016: Add constant memory arrays to store parameters for all projections to be analyzed during a single kernel call
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
//
//      Function:       kernelPixelBackprojectionFDK
//
//      Description:    Main FDK backprojection kernel
//______________________________________________________________________________

static inline void kernelPixelBackprojectionFDK(const Geometry& geo, float* image,const int currProjSetNumber, const int totalNoOfProjections, const float* tex,
        const Point3D* projParamsArrayDev, const float* projSinCosArrayDev, sycl::id<3> id)
{
    
    // Old kernel call signature:
    // kernelPixelBackprojectionFDK<<<grid,block>>>(geo,dimage,i,deltaX,deltaY,deltaZ,xyzOrigin,offOrig,offDetec,sinalpha,cosalpha);
    // We just read in most of the params from the constant memory instead of getting them from the param list.
    // This is because we now have MANY params, since single kernel processes more than one projection!
    /* __global__ void kernelPixelBackprojectionFDK(const Geometry geo,
     * float* image,
     * const int indAlpha,
     * const Point3D deltaX ,
     * const Point3D deltaY,
     * const Point3D deltaZ,
     * const Point3D xyzOrigin,
     * const Point3D xyzOffset,
     * const Point3D uv0Offset,
     * const float sinalpha,
     * const float cosalpha){
     */
    unsigned long long indY = id[1];
    unsigned long long indX = id[2];
    // unsigned long startIndZ = blockIdx.z * blockDim.z + threadIdx.z;  // This is only STARTING z index of the column of voxels that the thread will handle
    unsigned long long startIndZ = id[0] * VOXELS_PER_THREAD;  // This is only STARTING z index of the column of voxels that the thread will handle
    //Make sure we don't go out of bounds
    if (indX>=geo.nVoxelX || indY>=geo.nVoxelY || startIndZ>=geo.nVoxelZ)
        return;
    
    // We'll keep a local auxiliary array of values of a column of voxels that this thread will update
    // Zero-initialised: voxels of the column beyond the image are computed too (no per-voxel check, see below), never stored.
    float voxelColumn[VOXELS_PER_THREAD] = {0};
    
    // First we need to copy the curent 3D volume values from the column to our auxiliary array so that we can then
    // work on them (update them by computing values from multiple projections) locally - avoiding main memory reads/writes
    
    unsigned long colIdx;
    // The voxel's indices as floats, converted once per thread (they were converted for every voxel and projection).
    const float fIndX=(float)indX, fIndY=(float)indY, fStartIndZ=(float)startIndZ;
    for(colIdx=0; colIdx<VOXELS_PER_THREAD; colIdx++)
    {
        unsigned long long indZ = startIndZ + colIdx;
        // If we are out of bounds, break the loop. The voxelColumn array will be updated partially, but it is OK, because we won't
        // be trying to copy the out of bounds values back to the 3D volume anyway (bounds checks will be done in the final loop where the updated values go back to the main volume)
        if(indZ>=geo.nVoxelZ)
            break;   // break the loop.
        
        unsigned long long idx =indZ*(unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY+indY*(unsigned long long)geo.nVoxelX + indX;
        voxelColumn[colIdx] = image[idx];   // Read the current volume value that we'll update by computing values from MULTIPLE projections (not just one)
        // We'll be updating the local (register) variable, avoiding reads/writes from the slow main memory.
    }  // END copy 3D volume voxels to local array
    
    // Now iterate through projections
    for(unsigned long projNumber=0; projNumber<PROJ_PER_KERNEL; projNumber++)
    {
        // Get the current parameters from parameter arrays in constant memory.
        unsigned long indAlpha = currProjSetNumber*PROJ_PER_KERNEL+projNumber;  // This is the ABSOLUTE projection number in the projection array
        
        // Our currImageVal will be updated by hovewer many projections we had left in the "remainder" - that's OK.
        if(indAlpha>=totalNoOfProjections)
            break;
        
        // The projection's parameters, folded on the host (see fdkKernelParams): the source-to-voxel vector is
        // c0 + i*deltaX + j*deltaY + k*deltaZ for voxel (i,j,k), in the scaled coordinate system rotated with the
        // projection (detector rotation, offsets and COR included).
        Point3D deltaX = projParamsArrayDev[6*projNumber];  // 6*projNumber because we have 6 Point3D values per projection
        Point3D deltaY = projParamsArrayDev[6*projNumber+1];
        Point3D deltaZ = projParamsArrayDev[6*projNumber+2];
        Point3D c0     = projParamsArrayDev[6*projNumber+3];
        Point3D tuv    = projParamsArrayDev[6*projNumber+4];  // (DSO-DSD-S.x, S.y+nDetecU/2, S.z+nDetecV/2)
        Point3D wden   = projParamsArrayDev[6*projNumber+5];  // the FDK weight's denominator: wden.x+i*wden.y+j*wden.z
        
        // The vector from the source to the first voxel of the column; then stepped by deltaZ (at most
        // VOXELS_PER_THREAD-1 steps).
        float vectX=c0.x+fIndX*deltaX.x+fIndY*deltaY.x+fStartIndZ*deltaZ.x;
        float vectY=c0.y+fIndX*deltaX.y+fIndY*deltaY.y+fStartIndZ*deltaZ.y;
        float vectZ=c0.z+fIndX*deltaX.z+fIndY*deltaY.z+fStartIndZ*deltaZ.z;
        const float tNum=tuv.x, u0=tuv.y, v0=tuv.z;
        
        // The FDK weight, (DSO/(DSO+realy*sin(alpha)-realx*cos(alpha)))^2, depends on the voxel's x and y only: the
        // same for the whole column.
        const float w=wden.x+fIndX*wden.y+fIndY*wden.z;
        const float weight=1.0f/(w*w);
        // indAlpha is the ABSOLUTE number of projection in the projection array (NOT the current number of projection set!)
        const float fAlpha=(float)indAlpha+0.5f;
        
        // Now iterate through Z in our voxel column FOR A GIVEN PROJECTION
        // No bounds check per voxel: the voxels of a column that lie beyond the image (in the last block in z) are
        // computed too, harmlessly (a bounds-checked fetch), and never stored (the final loop checks).
        for(colIdx=0; colIdx<VOXELS_PER_THREAD; colIdx++)
        {
            // Get the coordinates in the detector UV where the mid point of the voxel is projected.
            float t=tNum/vectX;
            float u=vectY*t+u0;
            float v=vectZ*t+v0;
            
            // Get Value in the computed (U,V) and multiply by the corresponding weight.
#if IS_FOR_MATLAB_TIGRE
            voxelColumn[colIdx]+=tigreFetchLinear3D(tex, geo.nDetecV, geo.nDetecU, totalNoOfProjections, v, u ,fAlpha)*weight;
#else
            voxelColumn[colIdx]+=tigreFetchLinear3D(tex, geo.nDetecU, geo.nDetecV, totalNoOfProjections, u, v ,fAlpha)*weight;
#endif
            vectX+=deltaZ.x; vectY+=deltaZ.y; vectZ+=deltaZ.z;
        }  // END iterating through column of voxels
        
    }  // END iterating through multiple projections
    
    // And finally copy the updated local voxelColumn array back to our 3D volume (main memory)
    for(colIdx=0; colIdx<VOXELS_PER_THREAD; colIdx++)
    {
        unsigned long long indZ = startIndZ + colIdx;
        // If we are out of bounds, break the loop. The voxelColumn array will be updated partially, but it is OK, because we won't
        // be trying to copy the out of bounds values back to the 3D volume anyway (bounds checks will be done in the final loop where the values go to the main volume)
        if(indZ>=geo.nVoxelZ)
            break;   // break the loop.
        
        unsigned long long idx =indZ*(unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY+indY*(unsigned long long)geo.nVoxelX + indX;
        image[idx] = voxelColumn[colIdx];   // Read the current volume value that we'll update by computing values from MULTIPLE projections (not just one)
        // We'll be updating the local (register) variable, avoiding reads/writes from the slow main memory.
        // According to references (Papenhausen), doing = is better than +=, since += requires main memory read followed by a write.
        // We did all the reads into the local array at the BEGINNING of this kernel. According to Papenhausen, this type of read-write split is
        // better for avoiding memory congestion.
    }  // END copy updated voxels from local array to our 3D volume
    
}  // END kernelPixelBackprojectionFDK


static void fdkKernelParams(const Geometry& geo, int i, float sinalpha, float cosalpha, Point3D* params);  // see its definition

int voxel_backprojection(float  *  projections, Geometry geo, float* result,float const * const alphas, int nalpha, const GpuIds& gpuids, bool gpuData)
{
    // GPU arrays (#857): the pointers would be CUDA device memory, which a SYCL device
    // cannot read. tigre.Ax and tigre.Atb pass such arrays through the host instead.
    if (gpuData) {
        mexErrMsgIdAndTxt("Atb:Voxel_backprojection:GPUdata","GPU arrays (PyTorch CUDA tensors, CuPy arrays) are not supported by the SYCL build: pass host (NumPy) arrays.\n");
    }
    // Prepare for MultiGPU
    int deviceCount = gpuids.GetLength();
    if (deviceCount == 0) {
        mexErrMsgIdAndTxt("Atb:Voxel_backprojection:GPUselect","There are no available SYCL device(s)\n");
    }
    if (!gpuids.AreEqualDevices()) {
        mexWarnMsgIdAndTxt("Atb:Voxel_backprojection:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
    }
    int dev;
    // If an error raises (Python build) before the end of this function, onError
    // releases what it holds; see tigre_release_on_error.hpp.
    tigre::ReleaseOnError onError;
    try {
        // Same image/projection split rule as the CUDA code
        unsigned int split_image;
        unsigned int split_projections;
        bool one_upload=false;
        splitCTbackprojection(gpuids,geo,nalpha,&split_image,&split_projections,&one_upload);

        size_t num_pixels_proj = (size_t)geo.nDetecU*(size_t)geo.nDetecV;
        size_t mem_GPU_global;
        checkFreeMemory(gpuids, &mem_GPU_global);
        std::vector<int> devs(deviceCount);
        for (dev = 0; dev < deviceCount; dev++) devs[dev] = gpuids[dev];

        // Allocate the image slab and the projection/parameter buffers on every device. If that
        // fails although the estimate above fits, split the projections or the image in more
        // pieces and try again, down to one slice and one projection, as the CUDA code does.
        Geometry* geoArray;
        unsigned int proj_split_size, params_size;
        size_t num_voxels_img;
        std::vector<float*> dimage(deviceCount), dprojection(deviceCount), dSinCos(deviceCount);
        std::vector<Point3D*> dParams(deviceCount);
        std::vector<std::vector<Point3D> > projParamsArrayHost;
        std::vector<std::vector<float> > projSinCosArrayHost;
        while (true) {
            geoArray = (Geometry*)malloc((size_t)split_image*deviceCount*sizeof(Geometry));
            createGeoArray(split_image*deviceCount,geo,geoArray,nalpha);
            num_voxels_img = (size_t)geoArray[0].nVoxelX*(size_t)geoArray[0].nVoxelY*(size_t)geoArray[0].nVoxelZ;
            // The projections of a split are all copied to the device at once: make the
            // splits small enough to fit next to the image slab.
            size_t mem_slab=num_voxels_img*sizeof(float);
            size_t mem_avail=(mem_GPU_global>mem_slab) ? (mem_GPU_global-mem_slab)/2 : 0;
            size_t max_split=std::max((size_t)PROJ_PER_KERNEL, (mem_avail/(num_pixels_proj*sizeof(float)))/PROJ_PER_KERNEL*PROJ_PER_KERNEL);
            unsigned int split_projections_try=std::max(split_projections, (unsigned int)((nalpha+max_split-1)/max_split));
            proj_split_size=(nalpha+split_projections_try-1)/split_projections_try;
            params_size=((proj_split_size+PROJ_PER_KERNEL-1)/PROJ_PER_KERNEL)*PROJ_PER_KERNEL;
            const size_t bytes[4] = {
                num_voxels_img*sizeof(float),
                (size_t)proj_split_size*num_pixels_proj*sizeof(float),
                (size_t)6*params_size*sizeof(Point3D),
                (size_t)5*params_size*sizeof(float)
            };
            std::vector<void*> ptrs((size_t)deviceCount*4);
            if (tigre::tryMallocBuffers(devs.data(), deviceCount, 4, bytes, ptrs.data())) {
                split_projections=split_projections_try;
                for (dev = 0; dev < deviceCount; dev++) {
                    dimage[dev] = (float*)ptrs[(size_t)dev*4+0];
                    dprojection[dev] = (float*)ptrs[(size_t)dev*4+1];
                    dParams[dev] = (Point3D*)ptrs[(size_t)dev*4+2];
                    dSinCos[dev] = (float*)ptrs[(size_t)dev*4+3];
                }
                projParamsArrayHost.assign(deviceCount, std::vector<Point3D>((size_t)6*params_size));
                projSinCosArrayHost.assign(deviceCount, std::vector<float>((size_t)5*params_size));
                break;
            }
            const unsigned int piece = geoArray[0].nVoxelZ;
            freeGeoArray(split_image*deviceCount, geoArray);
            if (proj_split_size>1) {
                split_projections=split_projections_try+1;
                continue;
            }
            if (piece<=1 || (unsigned long long)split_image*deviceCount>=(unsigned long long)geo.nVoxelZ)
                tigreNotEnoughGpuMemory("Atb:voxel_backprojection:GPUmemory","Backprojection",((size_t)geo.nVoxelX*geo.nVoxelY+num_pixels_proj)*sizeof(float),tigreSyclFreeMemory(gpuids[0]));
            while ((unsigned long long)split_image*deviceCount<(unsigned long long)geo.nVoxelZ &&
                   (geo.nVoxelZ+split_image*deviceCount-1)/(split_image*deviceCount)>=piece)
                split_image++;
        }
        onError.add([geoArray, split_image, deviceCount]{ freeGeoArray(split_image*deviceCount, geoArray); });
        for (dev = 0; dev < deviceCount; dev++) {
            sycl::queue* q = &tigreSyclQueue(gpuids[dev]);
            onError.addQueue(q);
            float* image_d = dimage[dev];
            float* projection_d = dprojection[dev];
            Point3D* params_d = dParams[dev];
            float* sincos_d = dSinCos[dev];
            onError.add([image_d, projection_d, params_d, sincos_d, q]{
                sycl::free(image_d, *q); sycl::free(projection_d, *q);
                sycl::free(params_d, *q); sycl::free(sincos_d, *q);
            });
        }

        // Page-lock the projections and the result only when it is worth it (image split, or more than 2 devices)
        bool result_registered = false, projections_registered = false;
        if (tigre::worthPageLocking(tigre::hostRegisterSupported(devs.data(), deviceCount), gpuData, split_image, deviceCount)) {
            const size_t result_bytes = (size_t)geo.nVoxelX*(size_t)geo.nVoxelY*(size_t)geo.nVoxelZ*sizeof(float);
            const size_t proj_bytes = num_pixels_proj*(size_t)nalpha*sizeof(float);
            result_registered = tigre::hostRegister(result, result_bytes, devs.data(), deviceCount);
            if (result_registered) onError.add([result, devs]{ tigre::hostUnregister(result, devs.data(), (int)devs.size()); });
            projections_registered = tigre::hostRegister(projections, proj_bytes, devs.data(), deviceCount);
            if (projections_registered) onError.add([projections, devs]{ tigre::hostUnregister(projections, devs.data(), (int)devs.size()); });
        }

        for(unsigned int img_slice=0;img_slice<split_image;img_slice++){
            // Initialize the memory if its the first time.
            for (dev = 0; dev < deviceCount; dev++)
                tigreSyclQueue(gpuids[dev]).memset(dimage[dev],0,num_voxels_img*sizeof(float));

            for(unsigned int proj=0;proj<split_projections;proj++){
                unsigned int proj_start=proj*proj_split_size;
                if (proj_start>=(unsigned int)nalpha)
                    break;
                unsigned int nangles_split=std::min(proj_split_size,(unsigned int)nalpha-proj_start);
                unsigned int noOfKernelCalls = (nangles_split+PROJ_PER_KERNEL-1)/PROJ_PER_KERNEL;
                for (dev = 0; dev < deviceCount; dev++){
                    Geometry& geoSlab=geoArray[img_slice*deviceCount+dev];
                    if(geoSlab.nVoxelZ==0)
                        break;
                    sycl::queue& q = tigreSyclQueue(gpuids[dev]);
                    q.memcpy(dprojection[dev], &projections[(size_t)proj_start*num_pixels_proj], (size_t)nangles_split*num_pixels_proj*sizeof(float));
                    Point3D* projParamsArray=projParamsArrayHost[dev].data();
                    float* projSinCosArray=projSinCosArrayHost[dev].data();
                    for(unsigned int j=0; j<nangles_split; j++){
                        unsigned int currProjNumber_global=proj_start+j;
                        geoSlab.alpha=-alphas[currProjNumber_global*3];//we got 3 angles now.
                        geoSlab.theta=-alphas[currProjNumber_global*3+1];
                        geoSlab.psi  =-alphas[currProjNumber_global*3+2];
                        float sinalpha=sin(geoSlab.alpha);
                        float cosalpha=cos(geoSlab.alpha);
                        projSinCosArray[5*j]=sinalpha;  // 2*j because we have 2 float (sin or cos angle) values per projection
                        projSinCosArray[5*j+1]=cosalpha;
                        projSinCosArray[5*j+2]=geo.COR[currProjNumber_global];
                        projSinCosArray[5*j+3]=geo.DSD[currProjNumber_global];
                        projSinCosArray[5*j+4]=geo.DSO[currProjNumber_global];
                        fdkKernelParams(geoSlab,currProjNumber_global,sinalpha,cosalpha,&projParamsArray[6*j]);  // 6 Point3D per projection
                    }
                    q.memcpy(dParams[dev], projParamsArray, 6*params_size*sizeof(Point3D));
                    q.memcpy(dSinCos[dev], projSinCosArray, 5*params_size*sizeof(float));

                    const Geometry geoKernel=geoSlab;
                    float* image=dimage[dev];
                    const float* tex=dprojection[dev];
                    const int totalNoOfProjections=nangles_split;
                    sycl::range<3> range((geoKernel.nVoxelZ+VOXELS_PER_THREAD-1)/VOXELS_PER_THREAD, geoKernel.nVoxelY, geoKernel.nVoxelX);
                    for (unsigned int i=0; i<noOfKernelCalls; i++){
                        const Point3D* params=dParams[dev]+6*PROJ_PER_KERNEL*i;
                        const float* sincos=dSinCos[dev]+5*PROJ_PER_KERNEL*i;
                        q.parallel_for(range, [=](sycl::id<3> id){
                            kernelPixelBackprojectionFDK(geoKernel, image, i, totalNoOfProjections, tex, params, sincos, id);
                        });
                    }
                }
                // The host parameter arrays are reused by the next split
                for (dev = 0; dev < deviceCount; dev++)
                    tigreSyclQueue(gpuids[dev]).wait_and_throw();
            }
            // Copy the slabs back
            for (dev = 0; dev < deviceCount; dev++){
                Geometry& geoSlab=geoArray[img_slice*deviceCount+dev];
                size_t num_bytes_img_curr=(size_t)geoSlab.nVoxelX*(size_t)geoSlab.nVoxelY*(size_t)geoSlab.nVoxelZ*sizeof(float);
                size_t img_linear_idx_start=(size_t)geo.nVoxelX*(size_t)geo.nVoxelY*(size_t)geoArray[0].nVoxelZ*(size_t)(img_slice*deviceCount+dev);
                if (num_bytes_img_curr)
                    tigreSyclQueue(gpuids[dev]).memcpy(&result[img_linear_idx_start], dimage[dev], num_bytes_img_curr);
            }
            for (dev = 0; dev < deviceCount; dev++)
                tigreSyclQueue(gpuids[dev]).wait_and_throw();
        }
        onError.dismiss();  // from here on the function releases everything itself
        for (dev = 0; dev < deviceCount; dev++){
            sycl::queue& q = tigreSyclQueue(gpuids[dev]);
            sycl::free(dimage[dev], q);
            sycl::free(dprojection[dev], q);
            sycl::free(dParams[dev], q);
            sycl::free(dSinCos[dev], q);
        }
        freeGeoArray(split_image*deviceCount,geoArray);
        if (result_registered) tigre::hostUnregister(result, devs.data(), deviceCount);
        if (projections_registered) tigre::hostUnregister(projections, devs.data(), deviceCount);
    } catch (const sycl::exception& e) {
        mexErrMsgIdAndTxt("Atb:Voxel_backprojection:SYCL", e.what());
    }
    return 0;
}

void splitCTbackprojection(const GpuIds& gpuids, Geometry geo,int nalpha, unsigned int* split_image, unsigned int * split_projections, bool* one_upload){
    if (one_upload) *one_upload=false;  // SYCL copies each projection split into a buffer; no one-upload path.

    
    
    // We don't know if the devices are being used. lets check that. and only use the amount of memory we need.
    
    size_t mem_GPU_global;
    checkFreeMemory(gpuids, &mem_GPU_global);

    const int deviceCount = gpuids.GetLength();
        
    // Compute how much memory each of the relevant memory pieces need
    size_t mem_image=       (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY*(unsigned long long)geo.nVoxelZ*sizeof(float);
    size_t mem_proj=        (unsigned long long)geo.nDetecU*(unsigned long long)geo.nDetecV*sizeof(float);
    
    
    
    
    // Does everything fit in the GPU?
    
    if(mem_image/deviceCount+mem_proj*PROJ_PER_KERNEL*2<mem_GPU_global){
        // We only need to split if we have extra GPUs
        *split_image=1;
        *split_projections=1;
    }
    // We know we need to split, but:
    // Does all the image fit in the GPU, with some slack for a stack of projections??
    else
    {
        // As we can overlap memcpys from H2D of the projections, we should then minimize the amount of image splits.
        // Lets assume to start with that we only need 1 stack of PROJ_PER_KERNEL projections. The rest is for the image.
        // The smallest working set is one stack of projections and one image slice.
        const size_t mem_slice=(size_t)geo.nVoxelX*(size_t)geo.nVoxelY*sizeof(float);
        if (mem_GPU_global <= 2*mem_proj*PROJ_PER_KERNEL+mem_slice)
            tigreNotEnoughGpuMemory("Atb:voxel_backprojection:GPUmemory","Backprojection",2*mem_proj*PROJ_PER_KERNEL+mem_slice,mem_GPU_global);
        size_t mem_free=mem_GPU_global-2*mem_proj*PROJ_PER_KERNEL;
        
        *split_image=(mem_image/deviceCount+mem_free-1)/mem_free;
        // Now knowing how many splits we have for images, we can recompute how many slices of projections actually
        // fit on the GPU. Must be more than 0 obviously.
        
        mem_free=mem_GPU_global-(mem_image/deviceCount)/(*split_image); // NOTE: There is some rounding error, but its in the order of bytes, and we have 5% of GPU free jsut in case. We are safe
        
        
        *split_projections=(mem_proj*PROJ_PER_KERNEL*2+mem_free-1)/mem_free;
        
    }
}



//______________________________________________________________________________
//
//      Function:       createGeoArray
//
//      Description:    This code generates the geometries needed to split the image properly in
//                      cases where the entire image does not fit in the memory of the GPU
//______________________________________________________________________________

void createGeoArray(unsigned int image_splits, Geometry geo,Geometry* geoArray, unsigned int nangles){
    
    
    unsigned int  splitsize=(geo.nVoxelZ+image_splits-1)/image_splits;
    
    for(unsigned int sp=0;sp<image_splits;sp++){
        geoArray[sp]=geo;
        // All of them are splitsize, but the last ones, possibly fewer, or none: 5 slices on 4 GPUs
        // are pieces of 2, 2, 1 and 0 slices (the loops skip empty pieces). Counted in signed
        // arithmetic: nVoxelZ-splitsize*sp in unsigned wrapped around to a huge value.
        const long long rest=(long long)geo.nVoxelZ-(long long)splitsize*sp;
        geoArray[sp].nVoxelZ=(int)(rest<=0 ? 0 : (rest<(long long)splitsize ? rest : (long long)splitsize));
        geoArray[sp].sVoxelZ= geoArray[sp].nVoxelZ* geoArray[sp].dVoxelZ;
        
        // We need to redefine the offsets, as now each subimage is not aligned in the origin.
        geoArray[sp].offOrigZ=(float *)malloc(nangles*sizeof(float));
        for (unsigned int i=0;i<nangles;i++){
            geoArray[sp].offOrigZ[i]=geo.offOrigZ[i]-geo.sVoxelZ/2+sp*geoArray[0].sVoxelZ+geoArray[sp].sVoxelZ/2;
        }
    }
    
}
//______________________________________________________________________________
//
//      Function:       freeGeoArray
//
//      Description:    Frees the memory from the geometry array for multiGPU.
//______________________________________________________________________________
void freeGeoArray(unsigned int splits,Geometry* geoArray){
    for(unsigned int sp=0;sp<splits;sp++){
        free(geoArray[sp].offOrigZ);
    }
    free(geoArray);
}


//______________________________________________________________________________
//
//      double precision functions for rotating Point3Ddouble coordinates
//______________________________________________________________________________

void eulerZYZT(Geometry geo, Point3Ddouble* point){
    
    Point3Ddouble auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;

    // calculate sin and cos of 3 angles (used multiple times)
    double sin_alpha, cos_alpha, sin_theta, cos_theta, sin_psi, cos_psi;
    sin_alpha = sin((double)geo.alpha);
    cos_alpha = cos((double)geo.alpha);
    sin_theta = sin((double)geo.theta);
    cos_theta = cos((double)geo.theta);
    sin_psi = sin((double)geo.psi);
    cos_psi = cos((double)geo.psi);
    
    point->x = auxPoint.x*(cos_psi*cos_theta*cos_alpha-sin_psi*sin_alpha)
    +auxPoint.y*(-cos_psi*cos_theta*sin_alpha-sin_psi*cos_alpha)
    +auxPoint.z*cos_psi*sin_theta;
    point->y = auxPoint.x*(sin_psi*cos_theta*cos_alpha+cos_psi*sin_alpha)
    +auxPoint.y*(-sin_psi*cos_theta*sin_alpha+cos_psi*cos_alpha)
    +auxPoint.z*sin_psi*sin_theta;
    point->z =-auxPoint.x*sin_theta*cos_alpha
    +auxPoint.y*sin_theta*sin_alpha
    +auxPoint.z*cos_theta;
}

void rollPitchYawT(Geometry geo,int i, Point3Ddouble* point){

    Point3Ddouble auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;

    // calculate sin and cos of 3 angles (used multiple times)
    double sin_dRoll, cos_dRoll, sin_dPitch, cos_dPitch, sin_dYaw, cos_dYaw;
    sin_dRoll = sin((double)geo.dRoll[i]);
    cos_dRoll = cos((double)geo.dRoll[i]);
    sin_dPitch = sin((double)geo.dPitch[i]);
    cos_dPitch = cos((double)geo.dPitch[i]);
    sin_dYaw = sin((double)geo.dYaw[i]);
    cos_dYaw = cos((double)geo.dYaw[i]);
    
    point->x=cos_dRoll*cos_dPitch*auxPoint.x
            +sin_dRoll*cos_dPitch*auxPoint.y
            -sin_dPitch*auxPoint.z;
    
    point->y=(cos_dRoll*sin_dPitch*sin_dYaw - sin_dRoll*cos_dYaw)*auxPoint.x
            +(sin_dRoll*sin_dPitch*sin_dYaw + cos_dRoll*cos_dYaw)*auxPoint.y
            +cos_dPitch*sin_dYaw*auxPoint.z;
    
    point->z=(cos_dRoll*sin_dPitch*cos_dYaw + sin_dRoll*sin_dYaw)*auxPoint.x
            +(sin_dRoll*sin_dPitch*cos_dYaw - cos_dRoll*sin_dYaw)*auxPoint.y
            +cos_dPitch*cos_dYaw*auxPoint.z;
}

//______________________________________________________________________________
//
//      Function:       fdkKernelParams
//
//      Description:    The 6 Point3D that kernelPixelBackprojectionFDK reads for projection i (geo: this piece of
//                      the image, with geo.alpha/theta/psi set for the projection; sinalpha, cosalpha of geo.alpha).
//                      As fdkKernelParams in Common/CUDA/voxel_backprojection.cu, folded in double:
//                        [0..2] deltaX, deltaY, deltaZ: the change of the source-to-voxel vector per voxel index
//                        [3]    that vector at voxel (0,0,0): xyzOrigin - source, and COR (in detector pixels) in y
//                        [4]    (DSO-DSD-source.x, source.y+nDetecU/2, source.z+nDetecV/2): t=[4].x/vectX, u=vectY*t+[4].y
//                        [5]    the FDK weight's denominator, (DSO+realy*sin(alpha)-realx*cos(alpha))/DSO with realx,
//                               realy the voxel's centre, as w0 + i*wx + j*wy (the weight is 1/den^2)
//______________________________________________________________________________

static void fdkKernelParams(const Geometry& geo, int i, float sinalpha, float cosalpha, Point3D* params)
{
    Point3D deltaX, deltaY, deltaZ, xyzOrigin, source;
    computeDeltasCube(geo, i, &xyzOrigin, &deltaX, &deltaY, &deltaZ, &source);
    const double COR=geo.COR[i], DSO=geo.DSO[i], DSD=geo.DSD[i];
    Point3D c0, tuv, wden;
    c0.x=(float)((double)xyzOrigin.x-(double)source.x);
    c0.y=(float)((double)xyzOrigin.y-COR/(double)geo.dDetecU-(double)source.y);
    c0.z=(float)((double)xyzOrigin.z-(double)source.z);
    tuv.x=(float)(DSO-DSD-(double)source.x);
    tuv.y=(float)((double)source.y+(double)geo.nDetecU*0.5);
    tuv.z=(float)((double)source.z+(double)geo.nDetecV*0.5);
    const double rx0=-((double)geo.sVoxelX-(double)geo.dVoxelX)*0.5+(double)geo.offOrigX[i];
    const double ry0=-((double)geo.sVoxelY-(double)geo.dVoxelY)*0.5+(double)geo.offOrigY[i]+COR;
    wden.x=(float)((DSO+ry0*(double)sinalpha-rx0*(double)cosalpha)/DSO);
    wden.y=(float)(-(double)geo.dVoxelX*(double)cosalpha/DSO);
    wden.z=(float)( (double)geo.dVoxelY*(double)sinalpha/DSO);
    params[0]=deltaX;
    params[1]=deltaY;
    params[2]=deltaZ;
    params[3]=c0;
    params[4]=tuv;
    params[5]=wden;
}

//______________________________________________________________________________
//
//      Function:       computeDeltasCube
//
//      Description:    Computes relative increments for each projection (volume rotation).
//						Increments get passed to the backprojection kernel.
//______________________________________________________________________________

void computeDeltasCube(Geometry geo,int i, Point3D* xyzorigin, Point3D* deltaX, Point3D* deltaY, Point3D* deltaZ,Point3D* S)
{
    
    // initialize points with double precision
    Point3Ddouble P, Px,Py,Pz;

    // Get coords of Img(0,0,0)
    P.x=-(geo.sVoxelX/2-geo.dVoxelX/2)+geo.offOrigX[i];
    P.y=-(geo.sVoxelY/2-geo.dVoxelY/2)+geo.offOrigY[i];
    P.z=-(geo.sVoxelZ/2-geo.dVoxelZ/2)+geo.offOrigZ[i];
    
    // Get coords from next voxel in each direction
    Px.x=P.x+geo.dVoxelX;       Py.x=P.x;                Pz.x=P.x;
    Px.y=P.y;                   Py.y=P.y+geo.dVoxelY;    Pz.y=P.y;
    Px.z=P.z;                   Py.z=P.z;                Pz.z=P.z+geo.dVoxelZ;
    
    // Rotate image around X axis (this is equivalent of rotating the source and detector) RZ RY RZ
    eulerZYZT(geo,&P);
    eulerZYZT(geo,&Px);
    eulerZYZT(geo,&Py);
    eulerZYZT(geo,&Pz);
    
    //detector offset
    P.z =P.z-geo.offDetecV[i];            P.y =P.y-geo.offDetecU[i];
    Px.z =Px.z-geo.offDetecV[i];          Px.y =Px.y-geo.offDetecU[i];
    Py.z =Py.z-geo.offDetecV[i];          Py.y =Py.y-geo.offDetecU[i];
    Pz.z =Pz.z-geo.offDetecV[i];          Pz.y =Pz.y-geo.offDetecU[i];
    
    //Detector Roll pitch Yaw
    //
    // first, we need to offset everything so (0,0,0) is the center of the detector
    // Only X is required for that
    P.x=P.x+(geo.DSD[i]-geo.DSO[i]);
    Px.x=Px.x+(geo.DSD[i]-geo.DSO[i]);
    Py.x=Py.x+(geo.DSD[i]-geo.DSO[i]);
    Pz.x=Pz.x+(geo.DSD[i]-geo.DSO[i]);
    rollPitchYawT(geo,i,&P);
    rollPitchYawT(geo,i,&Px);
    rollPitchYawT(geo,i,&Py);
    rollPitchYawT(geo,i,&Pz);
    
    P.x=P.x-(geo.DSD[i]-geo.DSO[i]);
    Px.x=Px.x-(geo.DSD[i]-geo.DSO[i]);
    Py.x=Py.x-(geo.DSD[i]-geo.DSO[i]);
    Pz.x=Pz.x-(geo.DSD[i]-geo.DSO[i]);
    //Done for P, now source
    Point3Ddouble source;
    source.x=geo.DSD[i]; //already offseted for rotation
    source.y=-geo.offDetecU[i];
    source.z=-geo.offDetecV[i];
    rollPitchYawT(geo,i,&source);
    
    source.x=source.x-(geo.DSD[i]-geo.DSO[i]);//   source.y=source.y-auxOff.y;    source.z=source.z-auxOff.z;
    
//       mexPrintf("%f,%f,%f\n",source.x,source.y,source.z);
    // Scale coords so detector pixels are 1x1
    
    P.z =P.z /geo.dDetecV;                          P.y =P.y/geo.dDetecU;
    Px.z=Px.z/geo.dDetecV;                          Px.y=Px.y/geo.dDetecU;
    Py.z=Py.z/geo.dDetecV;                          Py.y=Py.y/geo.dDetecU;
    Pz.z=Pz.z/geo.dDetecV;                          Pz.y=Pz.y/geo.dDetecU;
    
    source.z=source.z/geo.dDetecV;                  source.y=source.y/geo.dDetecU;
    
    // get deltas of the changes in voxels
    deltaX->x=Px.x-P.x;   deltaX->y=Px.y-P.y;    deltaX->z=Px.z-P.z;
    deltaY->x=Py.x-P.x;   deltaY->y=Py.y-P.y;    deltaY->z=Py.z-P.z;
    deltaZ->x=Pz.x-P.x;   deltaZ->y=Pz.y-P.y;    deltaZ->z=Pz.z-P.z;
    
    // cast the results from the double precision calculations back to float
    *xyzorigin=P.to_float();
    *S=source.to_float();
}

void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global){
    // Free memory where the SYCL runtime reports it, otherwise the global memory
    // of the device (95%); never more than the largest single allocation.
    const int deviceCount = gpuids.GetLength();
    for (int dev = 0; dev < deviceCount; dev++){
        size_t mem = tigreSyclCheckFreeMemory(gpuids[dev],"Atb:Voxel_backprojection:GPUmemory");
        if(dev==0) *mem_GPU_global=mem;
        *mem_GPU_global=(mem<*mem_GPU_global)?mem:*mem_GPU_global;
    }
}

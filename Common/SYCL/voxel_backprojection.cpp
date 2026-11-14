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
    float voxelColumn[VOXELS_PER_THREAD];
    
    // First we need to copy the curent 3D volume values from the column to our auxiliary array so that we can then
    // work on them (update them by computing values from multiple projections) locally - avoiding main memory reads/writes
    
    unsigned long colIdx;
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
        
        Point3D deltaX = projParamsArrayDev[6*projNumber];  // 6*projNumber because we have 6 Point3D values per projection
        Point3D deltaY = projParamsArrayDev[6*projNumber+1];
        Point3D deltaZ = projParamsArrayDev[6*projNumber+2];
        Point3D xyzOrigin = projParamsArrayDev[6*projNumber+3];
        Point3D xyzOffset = projParamsArrayDev[6*projNumber+4];
        Point3D S = projParamsArrayDev[6*projNumber+5];
        
        float sinalpha = projSinCosArrayDev[5*projNumber];     // 2*projNumber because we have 2 float (sin or cos angle) values per projection
        float cosalpha = projSinCosArrayDev[5*projNumber+1];
        float COR = projSinCosArrayDev[5*projNumber+2];
        float DSD = projSinCosArrayDev[5*projNumber+3];
        float DSO = projSinCosArrayDev[5*projNumber+4];
        
        float auxCOR=COR/geo.dDetecU;
        // Now iterate through Z in our voxel column FOR A GIVEN PROJECTION
        for(colIdx=0; colIdx<VOXELS_PER_THREAD; colIdx++)
        {
            unsigned long long indZ = startIndZ + colIdx;
            
            // If we are out of bounds, break the loop. The voxelColumn array will be updated partially, but it is OK, because we won't
            // be trying to copy the out of bounds values anyway (bounds checks will be done in the final loop where the values go to the main volume)
            if(indZ>=geo.nVoxelZ)
                break;   // break the loop.
            
            // "XYZ" in the scaled coordinate system of the current point. The image is rotated with the projection angles.
            Point3D P;
            P.x=(xyzOrigin.x+indX*deltaX.x+indY*deltaY.x+indZ*deltaZ.x);
            P.y=(xyzOrigin.y+indX*deltaX.y+indY*deltaY.y+indZ*deltaZ.y)-auxCOR;
            P.z=(xyzOrigin.z+indX*deltaX.z+indY*deltaY.z+indZ*deltaZ.z);
            
            // This is the vector defining the line from the source to the Voxel
            float vectX,vectY,vectZ;
            vectX=(P.x -S.x);
            vectY=(P.y -S.y);
            vectZ=(P.z -S.z);
            
            // Get the coordinates in the detector UV where the mid point of the voxel is projected.
            float t=((DSO-DSD-S.x)/(vectX));
            float y,z;
            y=vectY*t+S.y;
            z=vectZ*t+S.z;
            float u,v;
            u=y+(float)geo.nDetecU*0.5f;
            v=z+(float)geo.nDetecV*0.5f;
            
            float weight;
            float realx,realy;
            realx=-(geo.sVoxelX-geo.dVoxelX)*0.5f  +indX*geo.dVoxelX   +xyzOffset.x;
            realy=-(geo.sVoxelY-geo.dVoxelY)*0.5f  +indY*geo.dVoxelY   +xyzOffset.y+COR;
            
            weight=((DSO+realy*sinalpha-realx*cosalpha)/(DSO));
            
            weight=(1.0f/(weight*weight));
            
            // Get Value in the computed (U,V) and multiply by the corresponding weight.
            // indAlpha is the ABSOLUTE number of projection in the projection array (NOT the current number of projection set!)
            
#if IS_FOR_MATLAB_TIGRE
            voxelColumn[colIdx]+=tigreFetchLinear3D(tex, geo.nDetecV, geo.nDetecU, totalNoOfProjections, v, u ,indAlpha+0.5f)*weight;
#else
            voxelColumn[colIdx]+=tigreFetchLinear3D(tex, geo.nDetecU, geo.nDetecV, totalNoOfProjections, u, v ,indAlpha+0.5f)*weight;
#endif
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


int voxel_backprojection(float  *  projections, Geometry geo, float* result,float const * const alphas, int nalpha, const GpuIds& gpuids)
{
    // Prepare for MultiGPU
    int deviceCount = gpuids.GetLength();
    if (deviceCount == 0) {
        mexErrMsgIdAndTxt("Atb:Voxel_backprojection:GPUselect","There are no available SYCL device(s)\n");
    }
    if (!gpuids.AreEqualDevices()) {
        mexWarnMsgIdAndTxt("Atb:Voxel_backprojection:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
    }
    int dev;
    try {
        // Same image/projection split rule as the CUDA code
        unsigned int split_image;
        unsigned int split_projections;
        splitCTbackprojection(gpuids,geo,nalpha,&split_image,&split_projections);

        // Each device gets a slab of the image along Z
        Geometry* geoArray=(Geometry*)malloc(split_image*deviceCount*sizeof(Geometry));
        createGeoArray(split_image*deviceCount,geo,geoArray,nalpha);

        size_t num_voxels_img = (size_t)geo.nVoxelX*(size_t)geo.nVoxelY*(size_t)geoArray[0].nVoxelZ;
        size_t num_pixels_proj = (size_t)geo.nDetecU*(size_t)geo.nDetecV;
        // The projections of a split are all copied to the device at once: make the
        // splits small enough to fit next to the image slab.
        size_t mem_GPU_global;
        checkFreeMemory(gpuids, &mem_GPU_global);
        size_t mem_slab=num_voxels_img*sizeof(float);
        size_t mem_avail=(mem_GPU_global>mem_slab) ? (mem_GPU_global-mem_slab)/2 : 0;
        size_t max_split=std::max((size_t)PROJ_PER_KERNEL, (mem_avail/(num_pixels_proj*sizeof(float)))/PROJ_PER_KERNEL*PROJ_PER_KERNEL);
        split_projections=std::max(split_projections, (unsigned int)((nalpha+max_split-1)/max_split));
        unsigned int proj_split_size=(nalpha+split_projections-1)/split_projections;
        unsigned int params_size=((proj_split_size+PROJ_PER_KERNEL-1)/PROJ_PER_KERNEL)*PROJ_PER_KERNEL;

        std::vector<float*> dimage(deviceCount), dprojection(deviceCount), dSinCos(deviceCount);
        std::vector<Point3D*> dParams(deviceCount);
        std::vector<std::vector<Point3D>> projParamsArrayHost(deviceCount, std::vector<Point3D>(6*params_size));
        std::vector<std::vector<float>> projSinCosArrayHost(deviceCount, std::vector<float>(5*params_size));
        for (dev = 0; dev < deviceCount; dev++){
            sycl::queue& q = tigreSyclQueue(gpuids[dev]);
            dimage[dev] = sycl::malloc_device<float>(num_voxels_img, q);
            dprojection[dev] = sycl::malloc_device<float>((size_t)proj_split_size*num_pixels_proj, q);
            dParams[dev] = sycl::malloc_device<Point3D>(6*params_size, q);
            dSinCos[dev] = sycl::malloc_device<float>(5*params_size, q);
            if (!dimage[dev] || !dprojection[dev] || !dParams[dev] || !dSinCos[dev])
                mexErrMsgIdAndTxt("Atb:Voxel_backprojection:malloc","Device memory allocation failed\n");
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
                        Point3D deltaX,deltaY,deltaZ,xyzOrigin, offOrig, source;
                        float sinalpha=sin(geoSlab.alpha);
                        float cosalpha=cos(geoSlab.alpha);
                        projSinCosArray[5*j]=sinalpha;  // 2*j because we have 2 float (sin or cos angle) values per projection
                        projSinCosArray[5*j+1]=cosalpha;
                        projSinCosArray[5*j+2]=geo.COR[currProjNumber_global];
                        projSinCosArray[5*j+3]=geo.DSD[currProjNumber_global];
                        projSinCosArray[5*j+4]=geo.DSO[currProjNumber_global];
                        computeDeltasCube(geoSlab,currProjNumber_global,&xyzOrigin,&deltaX,&deltaY,&deltaZ,&source);
                        offOrig.x=geo.offOrigX[currProjNumber_global];
                        offOrig.y=geo.offOrigY[currProjNumber_global];
                        offOrig.z=geoSlab.offOrigZ[currProjNumber_global];
                        projParamsArray[6*j]=deltaX;   // 6*j because we have 6 Point3D values per projection
                        projParamsArray[6*j+1]=deltaY;
                        projParamsArray[6*j+2]=deltaZ;
                        projParamsArray[6*j+3]=xyzOrigin;
                        projParamsArray[6*j+4]=offOrig;
                        projParamsArray[6*j+5]=source;
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
        for (dev = 0; dev < deviceCount; dev++){
            sycl::queue& q = tigreSyclQueue(gpuids[dev]);
            sycl::free(dimage[dev], q);
            sycl::free(dprojection[dev], q);
            sycl::free(dParams[dev], q);
            sycl::free(dSinCos[dev], q);
        }
        freeGeoArray(split_image*deviceCount,geoArray);
    } catch (const sycl::exception& e) {
        mexErrMsgIdAndTxt("Atb:Voxel_backprojection:SYCL", e.what());
    }
    return 0;
}

void splitCTbackprojection(const GpuIds& gpuids, Geometry geo,int nalpha, unsigned int* split_image, unsigned int * split_projections){
    
    
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
        // All of them are splitsize, but the last one, possible
        geoArray[sp].nVoxelZ=((sp+1)*splitsize<geo.nVoxelZ)?  splitsize:  std::max((int)geo.nVoxelZ-(int)(splitsize*sp),0);
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
    // SYCL has no portable "free memory" query: use the global memory of the
    // device (95%), capped by the largest single allocation it allows.
    const int deviceCount = gpuids.GetLength();
    for (int dev = 0; dev < deviceCount; dev++){
        size_t mem = tigreSyclMaxAlloc(gpuids[dev]);
        if(dev==0) *mem_GPU_global=mem;
        *mem_GPU_global=(mem<*mem_GPU_global)?mem:*mem_GPU_global;
    }
}

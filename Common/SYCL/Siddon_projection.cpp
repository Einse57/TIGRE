/*-------------------------------------------------------------------------
 *
 * SYCL functions for ray-voxel intersection based projection
 *
 * This file has the necessary fucntiosn to perform X-ray CBCT projection
 * operation given a geaometry, angles and image. It usesthe so-called
 * Jacobs algorithm to compute efficiently the length of the x-rays over
 * voxel space.
 *
 * CODE by       Ander Biguri
 *               Sepideh Hatamikia (arbitrary rotation)
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

#include <algorithm>
#include <vector>
#include <sycl/sycl.hpp>
#include "Siddon_projection.hpp"
#include "TIGRE_common.hpp"
#include "tigre_sycl.hpp"
#include <math.h>

#define MAXTREADS 1024
#define PROJ_PER_BLOCK 9
#define PIXEL_SIZE_BLOCK 9
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
     *    --->y   |   |        | /              |
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
    


// Parameters of the PROJ_PER_BLOCK projections of one kernel call (constant memory in the CUDA code)
struct ProjParams { Point3D data[4*PROJ_PER_BLOCK]; };

static inline void kernelPixelDetector(const Geometry& geo,
        float* detector,
        const int currProjSetNumber,
        const int totalNoOfProjections,
        const float* tex,
        const Point3D* projParamsArrayDev,
        sycl::id<3> id){
    
    
    unsigned long long u = id[2];
    unsigned long long v = id[1];
    unsigned long long projNumber=id[0];
    
    
    if (u>= geo.nDetecU || v>= geo.nDetecV || projNumber>=PROJ_PER_BLOCK)
        return;
    
#if IS_FOR_MATLAB_TIGRE
    size_t idx =  (size_t)(u * (unsigned long long)geo.nDetecV + v)+ projNumber*(unsigned long long)geo.nDetecV *(unsigned long long)geo.nDetecU ;
#else
    size_t idx =  (size_t)(v * (unsigned long long)geo.nDetecU + u)+ projNumber*(unsigned long long)geo.nDetecV *(unsigned long long)geo.nDetecU ;
#endif
    unsigned long indAlpha = currProjSetNumber*PROJ_PER_BLOCK+projNumber;  // This is the ABSOLUTE projection number in the projection array (for a given GPU)

    if(indAlpha>=totalNoOfProjections)
        return;
    
    Point3D uvOrigin = projParamsArrayDev[4*projNumber];  // 6*projNumber because we have 6 Point3D values per projection
    Point3D deltaU = projParamsArrayDev[4*projNumber+1];
    Point3D deltaV = projParamsArrayDev[4*projNumber+2];
    Point3D source = projParamsArrayDev[4*projNumber+3];
    
    /////// Get coordinates XYZ of pixel UV
    unsigned long pixelV = geo.nDetecV-v-1;
    unsigned long pixelU = u;
    Point3D pixel1D;
    pixel1D.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    pixel1D.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    pixel1D.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    ///////
    // Siddon's ray-voxel intersection, optimized as in doi=10.1.1.55.7516
    //////
    // Also called Jacobs algorithms
    Point3D ray;
    // vector of Xray
    ray.x=pixel1D.x-source.x;
    ray.y=pixel1D.y-source.y;
    ray.z=pixel1D.z-source.z;
    float eps=0.001;
    ray.x=(sycl::fabs(ray.x)<eps)? 0 : ray.x;
    ray.y=(sycl::fabs(ray.y)<eps)? 0 : ray.y; 
    ray.z=(sycl::fabs(ray.z)<eps)? 0 : ray.z; 
    // This variables are ommited because
    // bx,by,bz ={0,0,0}
    // dx,dy,dz ={1,1,1}
    // compute parameter values for x-ray parametric equation. eq(3-10)
    float axm,aym,azm;
    float axM,ayM,azM;
    // In the paper Nx= number of X planes-> Nvoxel+1
   
    axm=sycl::fmin(((-source.x)/(ray.x)),((geo.nVoxelX-source.x)/(ray.x)));
    aym=sycl::fmin(((-source.y)/(ray.y)),((geo.nVoxelY-source.y)/(ray.y)));
    azm=sycl::fmin(((-source.z)/(ray.z)),((geo.nVoxelZ-source.z)/(ray.z)));
    axM=sycl::fmax(((-source.x)/(ray.x)),((geo.nVoxelX-source.x)/(ray.x)));
    ayM=sycl::fmax(((-source.y)/(ray.y)),((geo.nVoxelY-source.y)/(ray.y)));
    azM=sycl::fmax(((-source.z)/(ray.z)),((geo.nVoxelZ-source.z)/(ray.z)));
    
    float am=sycl::fmax(sycl::fmax(axm,aym),azm);
    float aM=sycl::fmin(sycl::fmin(axM,ayM),azM);
    
    // line intersects voxel space ->   am<aM
    // The ray never enters the volume. RETURN HERE. Without it, execution
    // falls into the index block below, whose imin..kmax are only meaningful
    // for a ray that does enter. For a ray the eps guard has snapped to run
    // exactly along one voxel axis, with the source OUTSIDE the volume on
    // one of its zero axes, the two quotients for that axis share a sign, so
    // fminf and fmaxf collapse (axm == axM == -inf) and aM inherits -inf.
    // The other zero component then gives aM*ray = (-inf)*0 = NaN, Np is
    // NaN, and (unsigned long)NaN is 2^63 on CUDA: that thread iterates
    // ~9e18 times and the kernel never returns -- 100% SM, idle memory
    // controller, no error, no result. Measured: 2 rays in 261 million on an
    // odd x odd detector with an off-axis object; see
    // Python/tests/test_siddon_missing_ray.py and
    // Frontispiece/siddon_missed_ray_hang.png. A NEGATIVE Np does not do
    // this -- CUDA saturates negatives and -inf to 0, a zero-trip loop.
    if (am>=aM){
        detector[idx]=0;
        return;
    }

    // Compute max/min image INDEX for intersection eq(11-19)
    // Discussion about ternary operator in CUDA: https://stackoverflow.com/questions/7104384/in-cuda-why-is-a-b010-more-efficient-than-an-if-else-version
    float imin,imax,jmin,jmax,kmin,kmax;
    // for X
    if( source.x<pixel1D.x){
        imin=(am==axm)? 1.0f             : sycl::ceil(source.x+am*ray.x);
        imax=(aM==axM)? geo.nVoxelX      : sycl::floor(source.x+aM*ray.x);
    }else{
        imax=(am==axm)? geo.nVoxelX-1.0f : sycl::floor(source.x+am*ray.x);
        imin=(aM==axM)? 0.0f             : sycl::ceil(source.x+aM*ray.x);
    }
    // for Y
    if( source.y<pixel1D.y){
        jmin=(am==aym)? 1.0f             : sycl::ceil(source.y+am*ray.y);
        jmax=(aM==ayM)? geo.nVoxelY      : sycl::floor(source.y+aM*ray.y);
    }else{
        jmax=(am==aym)? geo.nVoxelY-1.0f : sycl::floor(source.y+am*ray.y);
        jmin=(aM==ayM)? 0.0f             : sycl::ceil(source.y+aM*ray.y);
    }
    // for Z
    if( source.z<pixel1D.z){
        kmin=(am==azm)? 1.0f             : sycl::ceil(source.z+am*ray.z);
        kmax=(aM==azM)? geo.nVoxelZ      : sycl::floor(source.z+aM*ray.z);
    }else{
        kmax=(am==azm)? geo.nVoxelZ-1.0f : sycl::floor(source.z+am*ray.z);
        kmin=(aM==azM)? 0.0f             : sycl::ceil(source.z+aM*ray.z);
    }
    
    // get intersection point N1. eq(20-21) [(also eq 9-10)]
    float ax,ay,az;
    ax=(source.x<pixel1D.x)?  ((imin-source.x)/(ray.x)) :  ((imax-source.x)/(ray.x));
    ay=(source.y<pixel1D.y)?  ((jmin-source.y)/(ray.y)) :  ((jmax-source.y)/(ray.y));
    az=(source.z<pixel1D.z)?  ((kmin-source.z)/(ray.z)) :  ((kmax-source.z)/(ray.z));
    
    // If its Infinite (i.e. ray is parallel to axis), make sure its positive
    ax=(sycl::isinf(ax))? sycl::fabs(ax) : ax;
    ay=(sycl::isinf(ay))? sycl::fabs(ay) : ay;
    az=(sycl::isinf(az))? sycl::fabs(az) : az;    
       
    
    // get index of first intersection. eq (26) and (19)
    long i,j,k;
    float aminc=sycl::fmin(sycl::fmin(ax,ay),az);
    i=(long)sycl::floor(source.x+ (aminc+am)*0.5f*ray.x);
    j=(long)sycl::floor(source.y+ (aminc+am)*0.5f*ray.y);
    k=(long)sycl::floor(source.z+ (aminc+am)*0.5f*ray.z);
    // Initialize
    float ac=am;
    //eq (28), unit anlges
    float axu,ayu,azu;
    axu=(1.0f/(sycl::fabs(ray.x)));
    ayu=(1.0f/(sycl::fabs(ray.y)));
    azu=(1.0f/(sycl::fabs(ray.z)));
    // eq(29), direction of update
    float iu,ju,ku;
    iu=(source.x< pixel1D.x)? 1.0f : -1.0f;
    ju=(source.y< pixel1D.y)? 1.0f : -1.0f;
    ku=(source.z< pixel1D.z)? 1.0f : -1.0f;
    
    float maxlength=sycl::sqrt(ray.x*ray.x*geo.dVoxelX*geo.dVoxelX+ray.y*ray.y*geo.dVoxelY*geo.dVoxelY+ray.z*ray.z*geo.dVoxelZ*geo.dVoxelZ);
    float sum=0.0f;
    long Np=(imax-imin+1)+(jmax-jmin+1)+(kmax-kmin+1); // Number of intersections
    // Go iterating over the line, intersection by intersection. If double point, no worries, 0 will be computed
    i+=0.5f;
    j+=0.5f;
    k+=0.5f;
    for (long ii=0;ii<Np;ii++){
        if (ax==aminc){
            sum+=(ax-ac)*tigreFetchPoint(tex, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, i, j, k);
            i=i+iu;
            ac=ax;
            ax+=axu;
        }else if(ay==aminc){
            sum+=(ay-ac)*tigreFetchPoint(tex, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, i, j, k);
            j=j+ju;
            ac=ay;
            ay+=ayu;
        }else if(az==aminc){
            sum+=(az-ac)*tigreFetchPoint(tex, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, i, j, k);
            k=k+ku;
            ac=az;
            az+=azu;
        }
        aminc=sycl::fmin(sycl::fmin(ax,ay),az);
    }
    detector[idx]=sum*maxlength;
}


int siddon_ray_projection(float* img, Geometry geo, float** result,float const * const angles,int nangles, const GpuIds& gpuids){
    // Prepare for MultiGPU
    int deviceCount = gpuids.GetLength();
    if (deviceCount == 0) {
        mexErrMsgIdAndTxt("Ax:Siddon_projection:GPUselect","There are no available SYCL device(s)\n");
    }
    if (!gpuids.AreEqualDevices()) {
        mexWarnMsgIdAndTxt("Ax:Siddon_projection:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
    }
    int dev;
    try {
        // Check the available device memory: same image split rule as the CUDA code.
        size_t mem_GPU_global;
        checkFreeMemory(gpuids, &mem_GPU_global);

        size_t mem_image=                 (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY*(unsigned long long)geo.nVoxelZ*sizeof(float);
        size_t mem_proj=                  (unsigned long long)geo.nDetecU*(unsigned long long)geo.nDetecV*sizeof(float);

        // Does everything fit in the device?
        const bool fits_in_memory = mem_image+2*PROJ_PER_BLOCK*mem_proj<mem_GPU_global;
        unsigned int splits=1;
        if (!fits_in_memory) {
            // Nope nope.
            // approx free memory we have. We already have left some extra 5% free for internal stuff
            // we need a second projection memory to combine multi-GPU stuff.
            size_t mem_free=mem_GPU_global-4*PROJ_PER_BLOCK*mem_proj;
            splits=mem_image/mem_free+1;// Ceil of the truncation
        }
        Geometry* geoArray = (Geometry*)malloc(splits*sizeof(Geometry));
        splitImage(splits,geo,geoArray,nangles);

        // Allocate the image (a slab of it if split) and one block of projections per device
        size_t num_voxels_split = (size_t)geoArray[0].nVoxelX*(size_t)geoArray[0].nVoxelY*(size_t)geoArray[0].nVoxelZ;
        size_t num_pixels_block = (size_t)PROJ_PER_BLOCK*(size_t)geo.nDetecU*(size_t)geo.nDetecV;
        std::vector<float*> dImage(deviceCount), dProjection(deviceCount);
        for (dev = 0; dev < deviceCount; dev++){
            sycl::queue& q = tigreSyclQueue(gpuids[dev]);
            dImage[dev] = sycl::malloc_device<float>(num_voxels_split, q);
            dProjection[dev] = sycl::malloc_device<float>(num_pixels_block, q);
            if (!dImage[dev] || !dProjection[dev])
                mexErrMsgIdAndTxt("Ax:Siddon_projection:malloc","Device memory allocation failed\n");
        }
        std::vector<float> partial(splits>1 ? num_pixels_block*deviceCount : 0);

        // Split the angles between devices
        int nangles_device=(nangles+deviceCount-1)/deviceCount;
        unsigned int noOfKernelCalls = (nangles_device+PROJ_PER_BLOCK-1)/PROJ_PER_BLOCK;  // We'll take care of bounds checking inside the loop if nalpha is not divisible by PROJ_PER_BLOCK

        ProjParams params;
        for (unsigned int sp=0;sp<splits;sp++){
            // Load the image (or the slab of it) on every device
            size_t linear_idx_start= (size_t)sp*(size_t)geoArray[0].nVoxelX*(size_t)geoArray[0].nVoxelY*(size_t)geoArray[0].nVoxelZ;
            size_t num_voxels_curr = (size_t)geoArray[sp].nVoxelX*(size_t)geoArray[sp].nVoxelY*(size_t)geoArray[sp].nVoxelZ;
            for (dev = 0; dev < deviceCount; dev++)
                tigreSyclQueue(gpuids[dev]).memcpy(dImage[dev], &img[linear_idx_start], num_voxels_curr*sizeof(float));

            const Geometry geoSplit = geoArray[sp];
            for (unsigned int i=0; i<noOfKernelCalls; i++) {
                std::vector<sycl::event> copies(deviceCount);
                std::vector<int> projection_this_block(deviceCount, 0);
                for (dev=0;dev<deviceCount;dev++){
                    unsigned int proj_start=i*PROJ_PER_BLOCK+dev*nangles_device;
                    if (proj_start>=nangles || i*PROJ_PER_BLOCK>=nangles_device)
                        continue;
                    projection_this_block[dev]=std::min(std::min(PROJ_PER_BLOCK, nangles_device-(int)(i*PROJ_PER_BLOCK)), nangles-(int)proj_start);
                    for(unsigned int j=0; j<PROJ_PER_BLOCK; j++){
                        unsigned int proj_global=(i*PROJ_PER_BLOCK+j)+dev*nangles_device;
                        if (proj_global>=nangles)
                            break;
                        if ((i*PROJ_PER_BLOCK+j)>=nangles_device)
                            break;
                        geoArray[sp].alpha=angles[proj_global*3];
                        geoArray[sp].theta=angles[proj_global*3+1];
                        geoArray[sp].psi  =angles[proj_global*3+2];
                        Point3D source, deltaU, deltaV, uvOrigin;
                        computeDeltas_Siddon(geoArray[sp],proj_global, &uvOrigin, &deltaU, &deltaV, &source);
                        params.data[4*j]=uvOrigin;
                        params.data[4*j+1]=deltaU;
                        params.data[4*j+2]=deltaV;
                        params.data[4*j+3]=source;
                    }
                    sycl::queue& q = tigreSyclQueue(gpuids[dev]);
                    float* detector = dProjection[dev];
                    const float* tex = dImage[dev];
                    const ProjParams p = params;
                    const int nangles_dev = nangles_device;
                    q.parallel_for(sycl::range<3>(PROJ_PER_BLOCK, geo.nDetecV, geo.nDetecU), [=](sycl::id<3> id){
                        kernelPixelDetector(geoSplit, detector, i, nangles_dev, tex, p.data, id);
                    });
                    float* dst = (sp==0) ? result[proj_start] : &partial[dev*num_pixels_block];
                    copies[dev] = q.memcpy(dst, detector, (size_t)projection_this_block[dev]*geo.nDetecV*geo.nDetecU*sizeof(float));
                }
                for (dev=0;dev<deviceCount;dev++){
                    if (!projection_this_block[dev])
                        continue;
                    copies[dev].wait_and_throw();
                    if (sp>0){
                        // Add the contribution of this slab of the image
                        unsigned int proj_start=i*PROJ_PER_BLOCK+dev*nangles_device;
                        float* acc = result[proj_start];
                        const float* add = &partial[dev*num_pixels_block];
                        size_t n = (size_t)projection_this_block[dev]*geo.nDetecV*geo.nDetecU;
                        for (size_t e=0; e<n; e++)
                            acc[e]+=add[e];
                    }
                }
            }
        }
        for (dev = 0; dev < deviceCount; dev++){
            sycl::queue& q = tigreSyclQueue(gpuids[dev]);
            q.wait_and_throw();
            sycl::free(dImage[dev], q);
            sycl::free(dProjection[dev], q);
        }
        freeGeoArray(splits,geoArray);
    } catch (const sycl::exception& e) {
        mexErrMsgIdAndTxt("Ax:Siddon_projection:SYCL", e.what());
    }
    return 0;
}


/* This code generates the geometries needed to split the image properly in
 * cases where the entire image does not fit in the memory of the GPU
 **/
void splitImage(unsigned int splits,Geometry geo,Geometry* geoArray, unsigned int nangles){
    
    unsigned long splitsize=(geo.nVoxelZ+splits-1)/splits;// ceil if not divisible
    for(unsigned int sp=0;sp<splits;sp++){
        geoArray[sp]=geo;
        // All of them are splitsize, but the last one, possible
        geoArray[sp].nVoxelZ=((sp+1)*splitsize<geo.nVoxelZ)?  splitsize:  geo.nVoxelZ-splitsize*sp;
        geoArray[sp].sVoxelZ= geoArray[sp].nVoxelZ* geoArray[sp].dVoxelZ;
        
        // We need to redefine the offsets, as now each subimage is not aligned in the origin.
        geoArray[sp].offOrigZ=(float *)malloc(nangles*sizeof(float));
        for (unsigned int i=0;i<nangles;i++){
            geoArray[sp].offOrigZ[i]=geo.offOrigZ[i]-geo.sVoxelZ/2+sp*geoArray[0].sVoxelZ+geoArray[sp].sVoxelZ/2;
        }
        
    }
    
}

/* This code precomputes The location of the source and the Delta U and delta V (in the warped space)
 * to compute the locations of the x-rays. While it seems verbose and overly-optimized,
 * it does saves about 30% of each of the kernel calls. Thats something!
 **/
void computeDeltas_Siddon(Geometry geo,int i, Point3D* uvorigin, Point3D* deltaU, Point3D* deltaV, Point3D* source){

    
    Point3D S;
    S.x=geo.DSO[i];
    S.y=0;
    S.z=0;
    
    //End point
    Point3D P,Pu0,Pv0;
    
    P.x  =-(geo.DSD[i]-geo.DSO[i]);   P.y  = geo.dDetecU*(-((double)geo.nDetecU/2.0)+0.5);       P.z  = geo.dDetecV*(((double)geo.nDetecV/2.0)-0.5);
    Pu0.x=0;                          Pu0.y= geo.dDetecU;                                    Pu0.z= 0;
    Pv0.x=0;                          Pv0.y= 0;                                              Pv0.z= geo.dDetecV*(-1);

    // Geometric transformations:
    // Now we have the Real world (OXYZ) coordinates of the bottom corner and its two neighbours.
    // The objective is to get a position of the detector in a coordinate system where:
    // 1-units are voxel size (in each direction can be different)
    // 2-The image has the its first voxel at (0,0,0)
    // 3-The image never rotates

    // To do that, we need to compute the "deltas" the detector, or "by how much
    // (in new xyz) does the voxels change when and index is added". To do that
    // several geometric steps needs to be changed

    //1.Roll,pitch,jaw
    // The detector can have a small rotation.
    // according to
    //"A geometric calibration method for cone beam CT systems" Yang K1, Kwan AL, Miller DF, Boone JM. Med Phys. 2006 Jun;33(6):1695-706.
    // Only the Z rotation will have a big influence in the image quality when they are small.
    // Still all rotations are supported

    // To roll pitch jaw, the detector has to be in centered in OXYZ.
    // NB: do not apply offsets to Pu0 and Pv0: they are directions, and are invariant through translations
    P.x=0;

    // Roll pitch yaw
    rollPitchYaw(geo,i,&P);
    rollPitchYaw(geo,i,&Pu0);
    rollPitchYaw(geo,i,&Pv0);
    //Now let's translate the points where they should be:
    // NB: do not apply offsets to Pu0 and Pv0: they are directions, and are invariant through translations
    P.x=P.x-(geo.DSD[i]-geo.DSO[i]);

    //1: Offset detector


    //S doesnt need to chagne


    //3: Rotate (around z)!
    Point3D Pfinal, Pfinalu0, Pfinalv0;
    Pfinal.x  =P.x;
    Pfinal.y  =P.y  +geo.offDetecU[i]; Pfinal.z  =P.z  +geo.offDetecV[i];
    Pfinalu0 = Pu0;
    Pfinalv0 = Pv0;

    eulerZYZ(geo,&Pfinal);
    eulerZYZ(geo,&Pfinalu0);
    eulerZYZ(geo,&Pfinalv0);
    eulerZYZ(geo,&S);

    //2: Offset image (instead of offseting image, -offset everything else)
    // NB: do not apply offsets to Pfinalu0 and Pfinalv0: they are directions, and are invariant through translations

    Pfinal.x  =Pfinal.x-geo.offOrigX[i];     Pfinal.y  =Pfinal.y-geo.offOrigY[i];     Pfinal.z  =Pfinal.z-geo.offOrigZ[i];
    S.x=S.x-geo.offOrigX[i];               S.y=S.y-geo.offOrigY[i];               S.z=S.z-geo.offOrigZ[i];

    // As we want the (0,0,0) to be in a corner of the image, we need to translate everything (after rotation);
    Pfinal.x  =Pfinal.x+geo.sVoxelX/2;      Pfinal.y  =Pfinal.y+geo.sVoxelY/2;          Pfinal.z  =Pfinal.z  +geo.sVoxelZ/2;
    S.x      =S.x+geo.sVoxelX/2;          S.y      =S.y+geo.sVoxelY/2;              S.z      =S.z      +geo.sVoxelZ/2;

    //4. Scale everything so dVoxel==1
    Pfinal.x  =Pfinal.x/geo.dVoxelX;      Pfinal.y  =Pfinal.y/geo.dVoxelY;        Pfinal.z  =Pfinal.z/geo.dVoxelZ;
    Pfinalu0.x=Pfinalu0.x/geo.dVoxelX;    Pfinalu0.y=Pfinalu0.y/geo.dVoxelY;      Pfinalu0.z=Pfinalu0.z/geo.dVoxelZ;
    Pfinalv0.x=Pfinalv0.x/geo.dVoxelX;    Pfinalv0.y=Pfinalv0.y/geo.dVoxelY;      Pfinalv0.z=Pfinalv0.z/geo.dVoxelZ;
    S.x      =S.x/geo.dVoxelX;          S.y      =S.y/geo.dVoxelY;            S.z      =S.z/geo.dVoxelZ;


    //mexPrintf("COR: %f \n",geo.COR[i]);
    //5. apply COR. Wherever everything was, now its offesetd by a bit
    // NB: do not apply offsets to Pfinalu0 and Pfinalv0: they are directions, and are invariant through translations
    double CORx, CORy;
    CORx=-geo.COR[i]*sin(geo.alpha)/geo.dVoxelX;
    CORy= geo.COR[i]*cos(geo.alpha)/geo.dVoxelY;
    Pfinal.x+=CORx;   Pfinal.y+=CORy;
    S.x+=CORx; S.y+=CORy;

    // return

    *uvorigin=Pfinal;

    *deltaU=Pfinalu0;
    *deltaV=Pfinalv0;
    
    *source=S;
}


#ifndef PROJECTION_HPP

float maxDistanceCubeXY(Geometry geo, float alpha,int i){
    ///////////
    // Compute initial "t" so we access safely as less as out of bounds as possible.
    //////////
    
    
    float maxCubX,maxCubY;
    // Forgetting Z, compute max distance: diagonal+offset
    maxCubX=(geo.sVoxelX/2+ abs(geo.offOrigX[i]))/geo.dVoxelX;
    maxCubY=(geo.sVoxelY/2+ abs(geo.offOrigY[i]))/geo.dVoxelY;
    
    return geo.DSO[i]/geo.dVoxelX-sqrt(maxCubX*maxCubX+maxCubY*maxCubY);
    
}
void rollPitchYaw(Geometry geo,int i, Point3D* point){
    Point3D auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;
    
    point->x=cos(geo.dRoll[i])*cos(geo.dPitch[i])*auxPoint.x
            +(cos(geo.dRoll[i])*sin(geo.dPitch[i])*sin(geo.dYaw[i]) - sin(geo.dRoll[i])*cos(geo.dYaw[i]))*auxPoint.y
            +(cos(geo.dRoll[i])*sin(geo.dPitch[i])*cos(geo.dYaw[i]) + sin(geo.dRoll[i])*sin(geo.dYaw[i]))*auxPoint.z;
    
    point->y=sin(geo.dRoll[i])*cos(geo.dPitch[i])*auxPoint.x
            +(sin(geo.dRoll[i])*sin(geo.dPitch[i])*sin(geo.dYaw[i]) + cos(geo.dRoll[i])*cos(geo.dYaw[i]))*auxPoint.y
            +(sin(geo.dRoll[i])*sin(geo.dPitch[i])*cos(geo.dYaw[i]) - cos(geo.dRoll[i])*sin(geo.dYaw[i]))*auxPoint.z;
    
    point->z=-sin(geo.dPitch[i])*auxPoint.x
            +cos(geo.dPitch[i])*sin(geo.dYaw[i])*auxPoint.y
            +cos(geo.dPitch[i])*cos(geo.dYaw[i])*auxPoint.z;
    
}
void eulerZYZ(Geometry geo, Point3D* point){
    Point3D auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;
    
    point->x=(+cos(geo.alpha)*cos(geo.theta)*cos(geo.psi)-sin(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-cos(geo.alpha)*cos(geo.theta)*sin(geo.psi)-sin(geo.alpha)*cos(geo.psi))*auxPoint.y+
            cos(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->y=(+sin(geo.alpha)*cos(geo.theta)*cos(geo.psi)+cos(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-sin(geo.alpha)*cos(geo.theta)*sin(geo.psi)+cos(geo.alpha)*cos(geo.psi))*auxPoint.y+
            sin(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->z=-sin(geo.theta)*cos(geo.psi)*auxPoint.x+
            sin(geo.theta)*sin(geo.psi)*auxPoint.y+
            cos(geo.theta)*auxPoint.z;
    
    
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
#endif

/*-------------------------------------------------------------------------
 *
 * PICCS minimization (minPICCS): steepest descent on
 *     (1-ratio)*TV(x) + ratio*TV(x - prior)
 *
 * This file has the PICCS maths: the step direction. The GPU bookkeeping
 * (several GPUs, images larger than the GPU memory, halos, the norm) is shared
 * with minTV: tv_gradient_descent (tv_gradient_descent.hpp, in GD_TV.cu), which
 * splits the prior like the image and calls piccsStepDirection for each piece.
 *
 * CODE by       Ander Biguri
 *
---------------------------------------------------------------------------
---------------------------------------------------------------------------
Copyright (c) 2015, University of Bath and CERN- European Organization for 
Nuclear Research
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
 ---------------------------------------------------------------------------

Contact: tigre.toolbox@gmail.com
Codes  : https://github.com/CERN/TIGRE
--------------------------------------------------------------------------- 
 */

#include <sycl/sycl.hpp>
#include "PICCS.hpp"
#include "tv_gradient_descent.hpp"

// The kernels of each TIGRE module share one SYCL runtime in a Python process, and the
// runtime finds kernels by name. A named namespace keeps these kernel names distinct
// from the identically named helpers of the other TV modules.
namespace tigre_piccs {

// u, or u - p when p is given.
static inline float value(const float* u, const float* p, size_t i){
    return p ? u[i]-p[i] : u[i];
}

// Backward differences at voxel (z,y,x) of u (or of u - p); 0 outside the image. In z the image is the slots
// [zlo,zhi) of the buffer (see TVPiece).
static inline void gradient(const float* u, const float* p, float* grad, long z, long y, long x,
                            long zlo, long zhi, long rows, long cols){
    if (z < zlo || z >= zhi || y >= rows || x >= cols)
        return;
    size_t size2d = (size_t)rows*cols;
    size_t idx = (size_t)z*size2d + y*cols + x;
    float uidx = value(u, p, idx);
    if (z - 1 >= zlo) grad[0] = uidx - value(u, p, idx - size2d);
    if (y - 1 >= 0)   grad[1] = uidx - value(u, p, idx - cols);
    if (x - 1 >= 0)   grad[2] = uidx - value(u, p, idx - 1);
}

// Gradient of the TV norm of u (or of u - p) at voxel (z,y,x).
static inline float tvGradient(const float* u, const float* p, long z, long y, long x,
                               long zlo, long zhi, long rows, long cols, float eps){
    float df[3]={0.f,0.f,0.f}, dfi[3]={0.f,0.f,0.f}, dfj[3]={0.f,0.f,0.f}, dfk[3]={0.f,0.f,0.f};
    gradient(u, p, df , z  , y  , x  , zlo, zhi, rows, cols);
    gradient(u, p, dfi, z  , y  , x+1, zlo, zhi, rows, cols);
    gradient(u, p, dfj, z  , y+1, x  , zlo, zhi, rows, cols);
    gradient(u, p, dfk, z+1, y  , x  , zlo, zhi, rows, cols);
    return (df[0]+df[1]+df[2])/(sycl::sqrt(df[0]*df[0]+df[1]*df[1]+df[2]*df[2])+eps)
          -dfi[2]/(sycl::sqrt(dfi[0]*dfi[0]+dfi[1]*dfi[1]+dfi[2]*dfi[2])+eps)
          -dfj[1]/(sycl::sqrt(dfj[0]*dfj[0]+dfj[1]*dfj[1]+dfj[2]*dfj[2])+eps)
          -dfk[0]/(sycl::sqrt(dfk[0]*dfk[0]+dfk[1]*dfk[1]+dfk[2]*dfk[2])+eps);
}

struct PiccsParams { float ratio; };

// PICCS step direction: (1-ratio)*grad TV(x) + ratio*grad TV(x - prior).
static void piccsStepDirection(const TVPiece& pc, const void* params){
    const float ratio = static_cast<const PiccsParams*>(params)->ratio;
    const float* x = pc.image;
    const float* prior = pc.extra[0];
    float* direction = pc.direction;
    const long depth = pc.depth, zlo = pc.zlo, zhi = pc.zhi, rows = pc.rows, cols = pc.cols;
    pc.stream->parallel_for(sycl::range<3>(depth, rows, cols), [=](sycl::id<3> id){
        const long iz = id[0], iy = id[1], ix = id[2];
        const float eps = 0.000001f;  // as PICCS always had (minTV uses 1e-8)
        direction[(size_t)iz*rows*cols + iy*cols + ix] =
            (1-ratio)*tvGradient(x, nullptr, iz, iy, ix, zlo, zhi, rows, cols, eps)
            +   ratio*tvGradient(x, prior  , iz, iy, ix, zlo, zhi, rows, cols, eps);
    });
}

}  // namespace tigre_piccs

void piccs_tv(const float* img,const float* prior, float* dst,float alpha,float ratio, const long* image_size, int maxIter, const GpuIds& gpuids){
    tigre_piccs::PiccsParams params = {ratio};
    const float* extra[1] = {prior};
    tv_gradient_descent(img, extra, 1, tigre_piccs::piccsStepDirection, &params, dst, alpha, image_size, maxIter, gpuids,
                        "PICCS minimization (minPICCS)");
}

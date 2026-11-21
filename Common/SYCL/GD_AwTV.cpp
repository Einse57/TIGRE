/*-------------------------------------------------------------------------
 *
 * Adaptive-weighted TV minimization (AwminTV): steepest descent on the
 * adaptive-weighted total variation of the image, using GPUs.
 *
 * This file has the AwTV maths: the step direction. The GPU bookkeeping
 * (several GPUs, images larger than the GPU memory, halos, the norm) is shared
 * with minTV and PICCS: tv_gradient_descent (tv_gradient_descent.hpp, in
 * GD_TV.cu), which calls awStepDirection for each piece of the image. Until
 * 3.3 this file had its own copy of that bookkeeping.
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

#include <sycl/sycl.hpp>
#include "GD_AwTV.hpp"
#include "tv_gradient_descent.hpp"

// The kernels of each TIGRE module share one SYCL runtime in a Python process, and the
// runtime finds kernels by name. A named namespace keeps these kernel names distinct
// from the identically named helpers of the other TV modules.
namespace tigre_gd_awtv {

// Backward differences at voxel (z,y,x) of the buffer; 0 outside the image. In z the image is the slots [zlo,zhi)
// of the buffer (see TVPiece in tv_gradient_descent.hpp).
static inline void gradient(const float* u, float* grad, long z, long y, long x,
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

// Gradient of the adaptive-weighted TV norm, each difference weighted by exp(-(difference/delta)^2).
static inline void gradientAwTV(const float* f, float* dftv,
        long depth, long zlo, long zhi, long rows, long cols, const float delta,
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

    float wx=sycl::exp(-(df[0]/delta)*(df[0]/delta));
    float wy=sycl::exp(-(df[1]/delta)*(df[1]/delta));
    float wz=sycl::exp(-(df[2]/delta)*(df[2]/delta));

    float wxi=sycl::exp(-(dfi[0]/delta)*(dfi[0]/delta));
    float wyi=sycl::exp(-(dfi[1]/delta)*(dfi[1]/delta));
    float wzi=sycl::exp(-(dfi[2]/delta)*(dfi[2]/delta));

    float wxj=sycl::exp(-(dfj[0]/delta)*(dfj[0]/delta));
    float wyj=sycl::exp(-(dfj[1]/delta)*(dfj[1]/delta));
    float wzj=sycl::exp(-(dfj[2]/delta)*(dfj[2]/delta));

    float wxk=sycl::exp(-(dfk[0]/delta)*(dfk[0]/delta));
    float wyk=sycl::exp(-(dfk[1]/delta)*(dfk[1]/delta));
    float wzk=sycl::exp(-(dfk[2]/delta)*(dfk[2]/delta));

    dftv[idx]=(wx*df[0]+wy*df[1]+wz*df[2])/(sycl::sqrt(wx*df[0] *df[0] +wy*df[1] *df[1] +wz*df[2] *df[2])+eps)
    -wzi*dfi[2]/(sycl::sqrt(wxi*dfi[0]*dfi[0]+wyi*dfi[1]*dfi[1]+wzi*dfi[2]*dfi[2]) +eps)
    -wyj*dfj[1]/(sycl::sqrt(wxj*dfj[0]*dfj[0]+wyj*dfj[1]*dfj[1]+wzj*dfj[2]*dfj[2]) +eps)
    -wxk*dfk[0]/(sycl::sqrt(wxk*dfk[0]*dfk[0]+wyk*dfk[1]*dfk[1]+wzk*dfk[2]*dfk[2]) +eps);
}

struct AwParams { float delta; };

static void awStepDirection(const TVPiece& pc, const void* params){
    const float delta = static_cast<const AwParams*>(params)->delta;
    const float* image = pc.image;
    float* direction = pc.direction;
    const long depth = pc.depth, zlo = pc.zlo, zhi = pc.zhi, rows = pc.rows, cols = pc.cols;
    pc.stream->parallel_for(sycl::range<3>(depth, rows, cols), [=](sycl::id<3> id){
        gradientAwTV(image, direction, depth, zlo, zhi, rows, cols, delta, id[2], id[1], id[0]);
    });
}

}  // namespace tigre_gd_awtv

void aw_pocs_tv(float* img,float* dst,float alpha,const long* image_size, int maxIter,const float delta, const GpuIds& gpuids){
    tigre_gd_awtv::AwParams params = {delta};
    tv_gradient_descent(img, NULL, 0, tigre_gd_awtv::awStepDirection, &params, dst, alpha, image_size, maxIter, gpuids,
                        "Adaptive-weighted total variation minimization (AwminTV)");
}

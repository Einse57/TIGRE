/*-------------------------------------------------------------------------
 *
 * SYCL functions for Steepest descend in POCS-type algorithms.
 *
 * This file will iteratively minimize by stepest descend the total variation 
 * of the input image, with the parameters given, using GPUs.
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







#define MAXTHREADS 1024

#include <math.h>
#include <sycl/sycl.hpp>
#include "PICCS.hpp"
#include "tigre_sycl.hpp"

// SYCL kernels
    static inline
            void gradient(const float* u, float* grad,
            long z, long y, long x,
            long depth, long rows, long cols)
    {
        // gradientTV() also asks for the x+1, y+1 and z+1 neighbours, which
        // fall outside the image on its last column, row and slice. Leave
        // grad at zero there instead of reading u[idx]: for the last slice
        // that read is past the end of the buffer (no halo is allocated here,
        // unlike GD_TV.cu), and the values only ever fed zero numerators.
        if (z >= depth || y >= rows || x >= cols)
            return;
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;

        float uidx = u[idx];

        if ( z - 1 >= 0 && z<depth) {
            grad[0] = (uidx-u[(z-1)*size2d + y*cols + x]) ;
        }
        
        if ( y - 1 >= 0 && y<rows){
            grad[1] = (uidx-u[z*size2d + (y-1)*cols + x]) ;
        }
        
        if ( x - 1 >= 0 && x<cols) {
            grad[2] = (uidx-u[z*size2d + y*cols + (x-1)]);
        }
    }
    
    static inline void gradientTV(const float* f, float* dftv,
            long depth, long rows, long cols,
            unsigned long x, unsigned long y, unsigned long z){
        unsigned long long idx = z * rows * cols + y * cols + x;
        if ( x >= cols || y >= rows || z >= depth )
            return;
        
        float df[3] ={0,0,0};
        float dfi[3]={0,0,0}; // dfi== \partial f_{i+1,j,k}
        float dfj[3]={0,0,0};
        float dfk[3]={0,0,0};
        gradient(f,df  ,z  ,y  ,x  , depth,rows,cols);
        gradient(f,dfi ,z  ,y  ,x+1, depth,rows,cols);
        gradient(f,dfj ,z  ,y+1,x  , depth,rows,cols);
        gradient(f,dfk ,z+1,y  ,x  , depth,rows,cols);
        float eps=0.000001; //% avoid division by zero
        dftv[idx]=(df[0]+df[1]+df[2])/(sycl::sqrt(df[0] *df[0] +df[1] *df[1] +df[2] *df[2])+eps)
        -dfi[2]/(sycl::sqrt(dfi[0]*dfi[0]+dfi[1]*dfi[1]+dfi[2]*dfi[2]) +eps)     // I wish I coudl precompute this, but if I do then Id need to recompute the gradient.
        -dfj[1]/(sycl::sqrt(dfj[0]*dfj[0]+dfj[1]*dfj[1]+dfj[2]*dfj[2]) +eps)
        -dfk[0]/(sycl::sqrt(dfk[0]*dfk[0]+dfk[1]*dfk[1]+dfk[2]*dfk[2]) +eps);
        
    }
    

    static void gradientTV(sycl::queue& q, const float* f, float* dftv,
            long depth, long rows, long cols){
        q.parallel_for(sycl::range<3>(depth, rows, cols), [=](sycl::id<3> id){
            gradientTV(f,dftv,depth,rows,cols,id[2],id[1],id[0]);
        });
    }
    static void multiplyArrayScalar(sycl::queue& q, float* vec, float scalar, const size_t n){
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ vec[i]*=scalar; });
    }
    static void substractArrays(sycl::queue& q, float* vec, float* vec2, const size_t n){
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ vec[i]-=vec2[i]; });
    }
    static void addArrays(sycl::queue& q, float* vec, float* vec2, const size_t n){
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ vec[i]+=vec2[i]; });
    }
    // Sum of squares of g_idata[0..n) into g_odata[0].
    static void reduceNorm2(sycl::queue& q, const float* g_idata, float* g_odata, size_t n){
        q.submit([&](sycl::handler& h){
            auto red = sycl::reduction(g_odata, 0.0f, sycl::plus<float>(),
                                       sycl::property::reduction::initialize_to_identity());
            h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> i, auto& sum){
                float v=g_idata[i];
                sum+=v*v;
            });
        });
    }
    
// main function
 void piccs_tv(const float* img,const float* prior, float* dst,float alpha,float ratio, const long* image_size, int maxIter, const GpuIds& gpuids){
    try {
        
     
        
    
            sycl::queue* q=&tigreSyclQueue(gpuids[0]);
            size_t total_pixels = image_size[0] * image_size[1]  * image_size[2] ;
            size_t mem_size = sizeof(float) * total_pixels;
        
            float *d_image,*d_prior,*d_dpiccsTV, *d_dimgTV,*d_aux_small,*d_aux_image, *d_norm2;
            // memory for image
            d_image=(float*)sycl::malloc_device(mem_size,*q);
            d_prior=(float*)sycl::malloc_device(mem_size,*q);

            q->memcpy(d_image,img,mem_size).wait();
            q->memcpy(d_prior,prior,mem_size).wait();
            // memory for df
            d_dimgTV=(float*)sycl::malloc_device(mem_size,*q);
            d_dpiccsTV=(float*)sycl::malloc_device(mem_size,*q);
            d_norm2=(float*)sycl::malloc_device(mem_size,*q);
            d_aux_image=(float*)sycl::malloc_device(mem_size,*q);
        
            // memory for L2norm auxiliar
            d_aux_small=(float*)sycl::malloc_device(sizeof(float)*(total_pixels + MAXTHREADS - 1) / MAXTHREADS,*q);
        
        
        
        
            // For the reduction
            float sumnorm2;


            for(unsigned int i=0;i<maxIter;i++){
            
                q->memcpy(d_aux_image,d_image,mem_size).wait();
    //             mexPrintf("Iteration %d\n",(int)i);

                // Compute the gradient of the TV norm
                gradientTV(*q,d_image,d_dimgTV,image_size[2], image_size[1],image_size[0]);
                q->wait();


                multiplyArrayScalar(*q,d_dimgTV,(1-ratio),   total_pixels);
                q->wait();

                substractArrays(*q,d_aux_image,d_prior, total_pixels);
                q->wait();
            
                gradientTV(*q,d_aux_image,d_dpiccsTV,image_size[2], image_size[1],image_size[0]);
                q->wait();

                multiplyArrayScalar(*q,d_dpiccsTV,ratio,   total_pixels);
                q->wait();
                
            
                addArrays(*q,d_dimgTV,d_dpiccsTV,total_pixels);
                q->wait();
                //NOMRALIZE via reduction
                q->memcpy(d_norm2,d_dimgTV,mem_size).wait();
                reduceNorm2(*q,d_norm2, d_aux_small, total_pixels);
                q->wait();
                q->memcpy(&sumnorm2, d_aux_small, sizeof(float)).wait();
    //             mexPrintf("alpha/sqrt(sumnorm2): %f\n",alpha/sqrt(sumnorm2));
                //MULTIPLY HYPERPARAMETER sqrt(sumnorm2)
                multiplyArrayScalar(*q,d_dimgTV,alpha/sqrt(sumnorm2),  total_pixels);
                q->wait();
                //SUBSTRACT GRADIENT
                substractArrays(*q,d_image,d_dimgTV, total_pixels);
                q->wait();
    //             mexPrintf("\n");
                sumnorm2=0;
            }
        
        
            q->memcpy(dst,d_image,mem_size).wait();
        
            sycl::free(d_image,*q);
            sycl::free(d_dimgTV,*q);
            sycl::free(d_dpiccsTV,*q);
            sycl::free(d_aux_image,*q);
            sycl::free(d_aux_small,*q);
            sycl::free(d_prior,*q);
            sycl::free(d_norm2,*q);


    
    } catch (const sycl::exception& e) {
        mexErrMsgIdAndTxt("PICCS:GPU", e.what());
    }
}
    

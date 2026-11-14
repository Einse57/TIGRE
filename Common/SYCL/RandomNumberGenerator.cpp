/*-------------------------------------------------------------------------
 *
 * SYCL functions for random number generator
 *
 * Adds noise of Poisson and normal distribution to the input.
 *
 * CODE by       Tomoyuki SADAKANE
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sycl/sycl.hpp>

#include "gpuUtils.hpp"
#include "RandomNumberGenerator.hpp"
#include "tigre_sycl.hpp"

// curand is not available in SYCL. Each element draws from its own
// Philox4x32-10 stream (counter = element index, key = fixed seed), so the
// result does not depend on the device or on how work is scheduled. The
// numbers differ from the CUDA build; the distributions are the same.
struct PhiloxStream {
    uint32_t ctr[4];
    uint32_t key[2];
    uint32_t out[4];
    int used;
    PhiloxStream(uint64_t idx, uint32_t seed){
        ctr[0]=(uint32_t)idx; ctr[1]=(uint32_t)(idx>>32); ctr[2]=0; ctr[3]=0;
        key[0]=seed; key[1]=0;
        used=4;
    }
    static inline uint32_t mulhi(uint32_t a, uint32_t b){
        return (uint32_t)(((uint64_t)a*(uint64_t)b)>>32);
    }
    void refill(){
        uint32_t c0=ctr[0], c1=ctr[1], c2=ctr[2], c3=ctr[3];
        uint32_t k0=key[0], k1=key[1];
        for (int r=0; r<10; ++r){
            uint32_t hi0=mulhi(0xD2511F53u,c0), lo0=0xD2511F53u*c0;
            uint32_t hi1=mulhi(0xCD9E8D57u,c2), lo1=0xCD9E8D57u*c2;
            c0=hi1^c1^k0; c1=lo1; c2=hi0^c3^k1; c3=lo0;
            k0+=0x9E3779B9u; k1+=0xBB67AE85u;
        }
        out[0]=c0; out[1]=c1; out[2]=c2; out[3]=c3;
        used=0;
        if (++ctr[2]==0) ++ctr[3];
    }
    // Uniform in (0,1].
    float uniform(){
        if (used==4) refill();
        return (float)((out[used++]>>8)+1u)*(1.0f/16777216.0f);
    }
    float normal(){
        float u1=uniform(), u2=uniform();
        return sycl::sqrt(-2.0f*sycl::log(u1))*sycl::cos(6.2831853071795864f*u2);
    }
    // log of the Poisson probability of k for mean lambda, written around
    // k-lambda so it keeps its accuracy in single precision for large lambda.
    static float logPoissonPmf(float k, float lambda){
        if (k<10.0f)
            return -lambda+k*sycl::log(lambda)-sycl::lgamma(k+1.0f);
        float d=k-lambda;
        float ik=1.0f/k;
        return d-k*sycl::log1p(d/lambda)-0.5f*sycl::log(6.2831853071795864f*k)-ik*(1.0f/12.0f-ik*ik*(1.0f/360.0f));
    }
    unsigned int poisson(float lambda){
        if (!(lambda>0.0f))
            return 0;
        if (lambda<10.0f){
            // Inversion by sequential search.
            float p=sycl::exp(-lambda), s=p, u=uniform();
            unsigned int k=0;
            while (u>s && k<1000u){
                ++k;
                p*=lambda/(float)k;
                s+=p;
            }
            return k;
        }
        // Transformed rejection with squeeze (W. Hormann, PTRS, 1993).
        float slam=sycl::sqrt(lambda);
        float b=0.931f+2.53f*slam;
        float a=-0.059f+0.02483f*b;
        float invalpha=1.1239f+1.1328f/(b-3.4f);
        float vr=0.9277f-3.6224f/(b-2.0f);
        while (true){
            float U=uniform()-0.5f;
            float V=uniform();
            float us=0.5f-sycl::fabs(U);
            float k=sycl::floor((2.0f*a/us+b)*U+lambda+0.43f);
            if (us>=0.07f && V<=vr)
                return (unsigned int)k;
            if (k<0.0f || (us<0.013f && V>us))
                continue;
            if (sycl::log(V)+sycl::log(invalpha)-sycl::log(a/(us*us)+b)<=logPoissonPmf(k,lambda))
                return (unsigned int)k;
        }
    }
};

static const uint32_t kSeed=1234;

static void generate(const float* pfIn, size_t uiLen, bool bGauss, float fGaussMu, float fGaussSigma, float* pfOut, int iDevice, const char* tag){
    try {
        sycl::queue& q=tigreSyclQueue(iDevice);
        float* d_pfIn =sycl::malloc_device<float>(uiLen, q);
        float* d_pfOut=sycl::malloc_device<float>(uiLen, q);
        q.memcpy(d_pfIn, pfIn, uiLen*sizeof(float));
        q.parallel_for(sycl::range<1>(uiLen), [=](sycl::id<1> id){
            size_t uiPos=id[0];
            PhiloxStream rng((uint64_t)uiPos, kSeed);
            unsigned int uiPoisson=rng.poisson(d_pfIn[uiPos]);
            if (bGauss){
                float fNormal=rng.normal()*fGaussSigma+fGaussMu;
                d_pfOut[uiPos]=fNormal+(float)uiPoisson;
            } else {
                d_pfOut[uiPos]=(float)uiPoisson;
            }
        });
        q.memcpy(pfOut, d_pfOut, uiLen*sizeof(float)).wait();
        sycl::free(d_pfIn, q);
        sycl::free(d_pfOut, q);
    } catch (const sycl::exception& e) {
        mexErrMsgIdAndTxt(tag, e.what());
    }
}

void poisson_1d(const float* pfIn, size_t uiLen, float* pfOut, const GpuIds& gpuids) {
    if (uiLen==0) return;
    generate(pfIn, uiLen, false, 0.0f, 0.0f, pfOut, gpuids.GetLength()>0 ? gpuids[0] : 0, "RandomNumberGenerator:poisson_1d");
}

void poisson_gaussian_1d(const float* pfIn,
                        size_t uiLen,
                        float fGaussMu,
                        float fGaussSigma,
                        float* pfOut,
                        GpuIds& gpuids)
{
    if (uiLen==0) return;
    generate(pfIn, uiLen, true, fGaussMu, fGaussSigma, pfOut, gpuids.GetLength()>0 ? gpuids[0] : 0, "RandomNumberGenerator:poisson_gaussian_1d");
}

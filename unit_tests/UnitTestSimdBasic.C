#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <SimdInterface.h>
#include <KokkosInterface.h>

#include "UnitTestUtils.h"

#include <limits>
#include <vector>

TEST(Simd, basic)
{
#if defined(STK_SIMD_AVX512) || defined(STK_SIMD_AVX) || defined(STK_SIMD_SSE)
    EXPECT_EQ(stk::simd::nfloats, 2*sierra::nalu::simdLen);
#endif
}

TEST(Simd, whichInstructions)
{
#if defined(STK_SIMD_AVX512)
   std::cout<<"STK_SIMD_AVX512";
#elif defined(STK_SIMD_AVX)
   std::cout<<"STK_SIMD_AVX";
#elif defined(STK_SIMD_SSE)
   std::cout<<"STK_SIMD_SSE";
#else
   std::cout<<"no simd instructions!"<<std::endl;
#endif
   std::cout<<", simdLen="<<sierra::nalu::simdLen<<std::endl;
}

typedef std::vector<double, non_std::AlignedAllocator<double,64> > aligned_vector;
template<typename T>
using AlignedVector = std::vector<T, non_std::AlignedAllocator<T,64> >;

void initialize(int N, aligned_vector& x, aligned_vector& y)
{
  for(int n=0; n<N; ++n) {
    x[n] = (rand()-0.5)/RAND_MAX;
    y[n] = (rand()-0.5)/RAND_MAX;
  }
}

TEST(Simd, stkMath)
{
  const int N = 512; // this is a multiple of the simd width
                     // if this is not true, the remainder 
                     // must be handled appropriately
  aligned_vector x(N);
  aligned_vector y(N);
  aligned_vector solution(N);
  
  initialize(N, x, y);
  
  for (int n=0; n < N; n+=sierra::nalu::simdLen) {
     const sierra::nalu::SimdDouble xl = stk::simd::load(&x[n]);
     const sierra::nalu::SimdDouble yl = stk::simd::load(&y[n]);
     sierra::nalu::SimdDouble zl = stk::math::abs(xl) * stk::math::exp(yl);
     stk::simd::store(&solution[n],zl);
  }   
  
  for (int n=0; n < N; ++n) {
     EXPECT_NEAR( std::abs(x[n]) * std::exp(y[n]), solution[n], 1.e-6 );
  }
} 

TEST(Simd, Views)
{
   const int N = 3;
   Kokkos::View<sierra::nalu::SimdDouble**> DoubleView("DoubleView",N,N);
   
   for(int i=0; i<N; ++i) {
      for(int j=0; j<N; ++j) {
         DoubleView(i,j) = 1.0*(i+j+1);
      }
   }

   for(int i=0; i<N; ++i) {
      for(int j=0; j<N; ++j) {
         sierra::nalu::SimdDouble& d = DoubleView(i,j);
         std::cout<<i<<","<<j<<": ";
         for(int k=0; k<sierra::nalu::simdLen; ++k) {
            std::cout<<stk::simd::get_data(d,k)<<",";
         }
         std::cout<<std::endl;
      }
   }

   sierra::nalu::SimdDouble& d = DoubleView(0,0);
   double* all = &d[0];
   for(int i=0; i<N*N*sierra::nalu::simdLen; ++i) {
     std::cout<<i<<": "<<all[i]<<std::endl;
   }
}


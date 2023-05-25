/*
 * Copyright (C) 2015, Simon Fuhrmann, Stepan Konrad
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 *
 * A helpful SSE/MMX overview.
 * Taken from: http://www.linuxjournal.com/content/
 *         ... introduction-gcc-compiler-intrinsics-vector-processing
 *
 * Compiler Options:
 * - X86/MMX/SSE1/2/3   -mfpmath=sse -mmmx -msse -msse2 -msse3
 * - ARM Neon           -mfpu=neon -mfloat-abi=softfp
 * - Freescale Altivec  -maltivec -mabi=altivec
 *
 * Include Files:
 * - arm_neon.h      ARM Neon types & intrinsics
 * - altivec.h       Freescale Altivec types & intrinsics
 * - mmintrin.h      X86 MMX
 * - xmmintrin.h     X86 SSE1
 * - emmintrin.h     X86 SSE2
 * - pmmintrin.h     X86 SSE3
 *
 * MMX/SSE Data Types:
 * - MMX:  __m64 64 bits of integers.
 * - SSE1: __m128 128 bits: four single precision floats.
 * - SSE2: __m128i 128 bits of packed integers, __m128d 128 bits: two doubles.
 *
 * Macros to check for availability:
 * - X86 MMX            __MMX__
 * - X86 SSE            __SSE__
 * - X86 SSE2           __SSE2__
 * - X86 SSE3           __SSE3__
 * - altivec functions  __VEC__
 * - neon functions     __ARM_NEON__
 */

#include <algorithm>
#include <iostream>
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3

#include "features/nearest_neighbor.h"

FEATURES_NAMESPACE_BEGIN

namespace
{
    /*
     * Signed and unsigned short inner product implementation. Stores the
     * largest and second largest inner product of query with elements.
     *
     * For SSE query and result should be 16 byte aligned.
     * Otherwise loading and storing values into/from registers is slow.
     * The dimension size must be divisible by 8, each __m128i register
     * can load 8 shorts = 16 bytes = 128 bit.
     */
    template <typename T>
    void
    short_inner_prod (T const* query,
        typename NearestNeighbor<T>::Result* result,
        T const* elements, int num_elements, int dimensions)
    {
#if ENABLE_SSE2_NN_SEARCH && defined(__SSE2__)
        /* Using a constant number reduces computation time by about 1/3. */
        int const dim_8 = dimensions / 8;
        __m128i const* descr_ptr = reinterpret_cast<__m128i const*>(elements);
        for (int descr_iter = 0; descr_iter < num_elements; ++descr_iter)
        {
            /* Compute dot product between query and candidate. */
            __m128i const* query_ptr = reinterpret_cast<__m128i const*>(query);
            __m128i reg_result = _mm_set1_epi16(0);
            for (int i = 0; i < dim_8; ++i, ++query_ptr, ++descr_ptr)
            {
                __m128i reg_query = _mm_load_si128(query_ptr);
                __m128i reg_subject = _mm_load_si128(descr_ptr);
                reg_result = _mm_add_epi16(reg_result,
                    _mm_mullo_epi16(reg_query, reg_subject));
            }
            T const* tmp = reinterpret_cast<T const*>(&reg_result);
            int inner_product = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            /* Check if new largest inner product has been found. */
            if (inner_product >= result->dist_2nd_best)
            {
                if (inner_product >= result->dist_1st_best)
                {
                    result->index_2nd_best = result->index_1st_best;
                    result->dist_2nd_best = result->dist_1st_best;
                    result->index_1st_best = descr_iter;
                    result->dist_1st_best = inner_product;
                }
                else
                {
                    result->index_2nd_best = descr_iter;
                    result->dist_2nd_best = inner_product;
                }
            }
        }
#else
        T const* descr_ptr = elements;
        for (int i = 0; i < num_elements; ++i)
        {
            int inner_product = 0;
            for (int i = 0; i < dimensions; ++i, ++descr_ptr)
                inner_product += query[i] * *descr_ptr;

            /* Check if new largest inner product has been found. */
            if (inner_product >= result->dist_2nd_best)
            {
                if (inner_product >= result->dist_1st_best)
                {
                    result->index_2nd_best = result->index_1st_best;
                    result->dist_2nd_best = result->dist_1st_best;
                    result->index_1st_best = i;
                    result->dist_1st_best = inner_product;
                }
                else
                {
                    result->index_2nd_best = i;
                    result->dist_2nd_best = inner_product;
                }
            }
        }
#endif
    }

    /*
     * Float inner product implementation. Stores the largest and
     * second largest inner product of query with elements.
     *
     * For SSE query and result should be 16 byte aligned.
     * Otherwise loading and storing values into/from registers is slow.
     * The dimension size must be divisible by 4, each __m128i register
     * can load 4 shorts = 16 bytes = 128 bit.
     */
    void
    float_inner_prod (float const* query,
        NearestNeighbor<float>::Result* result,
        float const* elements, int num_elements, int dimensions)
    {
#if ENABLE_SSE3_NN_SEARCH && defined(__SSE3__)
    /*
         * SSE inner product implementation.
         * Note that query and result should be 16 byte aligned.
         * Otherwise loading and storing values into/from registers is slow.
         * The dimension size must be divisible by 4, each __m128 register
         * can load 4 floats = 16 bytes = 128 bit.
         */

        __m128 const* descr_ptr = reinterpret_cast<__m128 const*>(elements);
        int const dim_4 = dimensions / 4;
        for (int descr_iter = 0; descr_iter < num_elements; ++descr_iter)
        {
            /* Compute dot product between query and candidate. */
            __m128 const* query_ptr = reinterpret_cast<__m128 const*>(query);
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < dim_4; ++i, ++query_ptr, ++descr_ptr)
                sum = _mm_add_ps(sum, _mm_mul_ps(*query_ptr, *descr_ptr));
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);

            /* Check if new largest inner product has been found. */
            float inner_product = _mm_cvtss_f32(sum);
            if (inner_product >= result->dist_2nd_best)
            {
                if (inner_product >= result->dist_1st_best)
                {
                    result->index_2nd_best = result->index_1st_best;
                    result->dist_2nd_best = result->dist_1st_best;
                    result->index_1st_best = descr_iter;
                    result->dist_1st_best = inner_product;
                }
                else
                {
                    result->index_2nd_best = descr_iter;
                    result->dist_2nd_best = inner_product;
                }
            }
        }
#else
        float const* descr_ptr = elements;
        for (int i = 0; i < num_elements; ++i)
        {
            float inner_product = 0.0f;
            for (int j = 0; j < dimensions; ++j, ++descr_ptr)
                inner_product += query[j] * *descr_ptr;

            /* Check if new largest inner product has been found. */
            if (inner_product >= result->dist_2nd_best)
            {
                if (inner_product >= result->dist_1st_best)
                {
                    result->index_2nd_best = result->index_1st_best;
                    result->dist_2nd_best = result->dist_1st_best;
                    result->index_1st_best = i;
                    result->dist_1st_best = inner_product;
                }
                else
                {
                    result->index_2nd_best = i;
                    result->dist_2nd_best = inner_product;
                }
            }
        }
#endif
    }

}

template <>
void
NearestNeighbor<short>::find (short const* query,
    NearestNeighbor<short>::Result* result) const  //T类型是short 类模板成员函数类外实现时需要加上模板参数列表：NearestNeighbor<unsigned short>::
{
    /* Result distances are shamelessly misused to store inner products. */
    //结果距离（可能是指计算出的某种距离度量）被错误地用于存储内积（向量之间的点积）。这种滥用可能导致数据的不正确解释或错误的计算结果。
    result->dist_1st_best = 0;
    result->dist_2nd_best = 0;
    result->index_1st_best = 0;
    result->index_2nd_best = 0;

    short_inner_prod<short>(query, result, this->elements,
        this->num_elements, this->dimensions);

    /*
     * Compute actual square distances.
     * The distance with 'signed char' vectors is: 2 * 127^2 - 2 * <Q, Ci>.
     * The maximum distance is (2*127)^2, which unfortunately does not fit
     * in a signed short. Therefore, the distance is clapmed at 127^2.
     */
    //这段注释解释了计算实际平方距离的过程。对于 'signed char' 类型的向量，距离的计算公式是 2 * 127^2 - 2 * <Q, Ci>。
    // 然而，最大距离 (2*127)^2 无法存储在有符号的 short 类型中，因此距离被限制在 127^2
    //该表达式的目的是将 (int)result->dist_1st_best 限制在 0 和 16129 之间的范围内，即返回最接近 (int)result->dist_1st_best 的值，同时不超过 16129。
    result->dist_1st_best = std::min(16129, std::max(0, (int)result->dist_1st_best));
    result->dist_2nd_best = std::min(16129, std::max(0, (int)result->dist_2nd_best));
    result->dist_1st_best = 32258 - 2 * result->dist_1st_best;
    result->dist_2nd_best = 32258 - 2 * result->dist_2nd_best;
}

template <>
void
NearestNeighbor<unsigned short>::find (unsigned short const* query,
    NearestNeighbor<unsigned short>::Result* result) const  //类模板成员函数类外实现时需要加上模板参数列表：NearestNeighbor<unsigned short>::
{
    /* Result distances are shamelessly misused to store inner products. */
    result->dist_1st_best = 0;
    result->dist_2nd_best = 0;
    result->index_1st_best = 0;
    result->index_2nd_best = 0;

    short_inner_prod<unsigned short>(query, result, this->elements,
        this->num_elements, this->dimensions);

    /*
     * Compute actual square distances.
     * The distance with 'unsigned char' vectors is: 2 * 255^2 - 2 * <Q, Ci>.
     * The maximum distance is (2*255)^2, which unfortunately does not fit
     * in a unsigned short. Therefore, the result distance is clapmed:
     * 2 * 255^2 - 2 * <Q, Ci> = 2 * (255^2 - <Q, Ci>) and (255^2 - <Q, Ci>)
     * is clamped to 32767 and then multiplied by 2.
     */

    //这段注释解释了计算实际平方距离的过程。对于 'unsigned char' 类型的向量，距离的计算公式是 2 * 255^2 - 2 * <Q, Ci>。
    // 然而，最大距离 (2*255)^2 无法存储在无符号的 short 类型中。
    // 因此，结果距离被限制为 2 * 255^2 - 2 * <Q, Ci> = 2 * (255^2 - <Q, Ci>)。其中 (255^2 - <Q, Ci>) 被限制为 32767，并且最后结果乘以 2。
    //由于计算结果需要存储在 'unsigned char' 类型中，而这个类型的范围是从 0 到 255，不包含负数。因此，2 * 255^2 的结果超出了该范围。
    // 为了限制结果距离在 'unsigned char' 类型的范围内，计算公式进行了调整，即 2 * 255^2 - 2 * <Q, Ci>。这样计算得到的结果范围是从 0 到 255。
    result->dist_1st_best = std::min(65025, (int)result->dist_1st_best);
    result->dist_2nd_best = std::min(65025, (int)result->dist_2nd_best);
    result->dist_1st_best = 65025 - result->dist_1st_best;
    result->dist_2nd_best = 65025 - result->dist_2nd_best;
    result->dist_1st_best = std::min(32767, (int)result->dist_1st_best) * 2;
    result->dist_2nd_best = std::min(32767, (int)result->dist_2nd_best) * 2;
}

template <>
void
NearestNeighbor<float>::find (float const* query,
    NearestNeighbor<float>::Result* result) const
{
    /* Result distances are shamelessly misused to store inner products. */
    result->dist_1st_best = 0.0f;
    result->dist_2nd_best = 0.0f;
    result->index_1st_best = 0;
    result->index_2nd_best = 0;

    float_inner_prod(query, result, this->elements,
        this->num_elements, this->dimensions);

    /*
     * Compute actual (square) distances.
     */
    result->dist_1st_best = std::max(0.0f, 2.0f - 2.0f * result->dist_1st_best);
    result->dist_2nd_best = std::max(0.0f, 2.0f - 2.0f * result->dist_2nd_best);
}

FEATURES_NAMESPACE_END

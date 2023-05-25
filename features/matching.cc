/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>

#include "math/algo.h"
#include "features/nearest_neighbor.h"
#include "features/matching.h"

FEATURES_NAMESPACE_BEGIN

/* 去除不一致的特征描述子 */
void
Matching::remove_inconsistent_matches (Matching::Result* matches) //Result* matches
{
    //matches_1_2 和 matches_2_1，分别表示从图像1到图像2的匹配索引和从图像2到图像1的匹配索引。
    //对于每个索引值 i，如果其对应的匹配索引值小于0，说明该匹配是不一致的，直接跳过。
    // 否则，检查该匹配索引在从图像2到图像1的匹配索引向量中对应的值是否等于 i。
    // 如果不相等，说明该匹配不是一致的，将从图像1到图像2的匹配索引置为-1，表示移除该不一致匹配。
    for (std::size_t i = 0; i < matches->matches_1_2.size(); ++i)
    {
        if (matches->matches_1_2[i] < 0)
            continue;
        if (matches->matches_2_1[matches->matches_1_2[i]] != (int)i)//类型转换 将变量 i 的值转换为 int 类型。
            matches->matches_1_2[i] = -1;
    }

    for (std::size_t i = 0; i < matches->matches_2_1.size(); ++i)
    {
        if (matches->matches_2_1[i] < 0)
            continue;
        if (matches->matches_1_2[matches->matches_2_1[i]] != (int)i)
            matches->matches_2_1[i] = -1;
    }
}

int
Matching::count_consistent_matches (Matching::Result const& matches) //Result const& matches
{
    int counter = 0;
    for (int i = 0; i < static_cast<int>(matches.matches_1_2.size()); ++i)
        //在循环体内，首先判断 matches.matches_1_2[i] 是否不等于 -1，表示该匹配是一致的。
        // 然后，再判断 matches.matches_2_1[matches.matches_1_2[i]] 是否等于 i，即从图像2到图像1的匹配索引是否等于 i。
        // 如果满足这两个条件，则表示存在一致的匹配，将 counter 的值增加1。
        if (matches.matches_1_2[i] != -1
            && matches.matches_2_1[matches.matches_1_2[i]] == i)
            counter++;
    return counter;
}

void
Matching::combine_results(Matching::Result const& sift_result,
    Matching::Result const& surf_result, Matching::Result* result) //Result对象 sift、surf  Result指针result（三个result）
{
    /////* Determine size of combined matching result. */ 两类1-2 、2-1的size结合
    std::size_t num_matches_1 = sift_result.matches_1_2.size()
        + surf_result.matches_1_2.size();
    std::size_t num_matches_2 = sift_result.matches_2_1.size()
        + surf_result.matches_2_1.size();

    ////* Combine results. */ 将两个向量的内容合并到 result->matches_1_2 向量中。
    result->matches_1_2.clear();
    result->matches_1_2.reserve(num_matches_1);//num_matches_1 = sift_result.matches_1_2.size()+ surf_result.matches_1_2.size();
    //向 result->matches_1_2 向量预留足够的空间以容纳 num_matches_1 个元素。这样做可以避免在插入元素时进行多次内存重新分配，提高性能。
    result->matches_1_2.insert(result->matches_1_2.end(),
        sift_result.matches_1_2.begin(), sift_result.matches_1_2.end());//将sift_result向量的内容插入到 result->matches_1_2 的末尾
    result->matches_1_2.insert(result->matches_1_2.end(),
        surf_result.matches_1_2.begin(), surf_result.matches_1_2.end());//将surf_result向量的内容插入到 result->matches_1_2 的末尾

    result->matches_2_1.clear();
    result->matches_2_1.reserve(num_matches_2);
    result->matches_2_1.insert(result->matches_2_1.end(),
        sift_result.matches_2_1.begin(), sift_result.matches_2_1.end());
    result->matches_2_1.insert(result->matches_2_1.end(),
        surf_result.matches_2_1.begin(), surf_result.matches_2_1.end());

    ////* Fix offsets. */  修正偏移量   sift->1_2  surf->2_1       /-------------仔细再看看-----2023/5/22-------/
    //这些偏移量表示 SURF 描述子的匹配结果在 result 对象中的起始位置。
    std::size_t surf_offset_1 = sift_result.matches_1_2.size();
    std::size_t surf_offset_2 = sift_result.matches_2_1.size();

    if (surf_offset_2 > 0)//表示在 result 对象中已经存在了 SURF 描述子的匹配结果。
        //该循环遍历 result->matches_1_2 向量中从 surf_offset_1 到末尾的元素，并对满足条件的元素进行修正。
        for (std::size_t i = surf_offset_1; i < result->matches_1_2.size(); ++i)
            //surf_offset_1 = sift_result.matches_1_2.size()
            //result->matches_1_2.size()=sift_result.matches_1_2.size()+ surf_result.matches_1_2.size();
            ////即result->matches_1_2.size()比surf_offset_1 多 surf_result.matches_1_2.size();

            if (result->matches_1_2[i] >= 0) //意味着在 result->matches_1_2 向量中的第 i 个位置存在一个有效的匹配索引。
                result->matches_1_2[i] += surf_offset_2;
    //surf_offset_2 表示了 SURF 描述子的匹配结果在 result 对象的 matches_2_1 向量中的起始位置之后的偏移量。
    // 因此，我们需要将这个偏移量添加到原始的匹配索引上，以确保匹配结果在 result 对象中的正确位置。
    //如果 result->matches_1_2[i] 大于等于 0，则将其增加 surf_offset_2。这样做是为了保持 SURF 描述子的匹配结果在 result 对象中的正确位置。
    //这个修正的目的是将 SURF 描述子的匹配结果与之前的匹配结果进行连接，以保持匹配结果的连续性和正确性。

    if (surf_offset_1 > 0)
        for (std::size_t i = surf_offset_2; i < result->matches_2_1.size(); ++i)
            if (result->matches_2_1[i] >= 0)
                result->matches_2_1[i] += surf_offset_1;
//surf_offset_1 表示了 SURF 描述子的匹配结果在 result 对象的 matches_1_2 向量中的起始位置之后的偏移量。
// 因此，我们需要将这个偏移量添加到原始的匹配索引上，以确保匹配结果在 result 对象中的正确位置。
//通过将 surf_offset_1 添加到 result->matches_2_1[i]，我们将该索引修正为 SURF 描述子的匹配结果在 result 对象的 matches_2_1 向量中的正确位置。
// 这样做是为了保持匹配结果的一致性，以便后续的处理和分析能够正确地使用和解释这些匹配结果。
}

FEATURES_NAMESPACE_END

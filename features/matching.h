/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SFM_MATCHING_HEADER
#define SFM_MATCHING_HEADER

#include <vector>
#include <limits>

#include "math/defines.h"
#include "features/defines.h"
#include "features/nearest_neighbor.h"

FEATURES_NAMESPACE_BEGIN

class Matching
{
public:
    /**
     * Feature matching options.
     * There are no default values, all fields must be initialized.
     */
    struct Options
    {
        /**
         * The length of the descriptor. Typically 128 for SIFT, 64 for SURF.
         */
        int descriptor_length;//描述子长度 128 for SIFT, 64 for SURF. 元素的维度

        /**
         * Requires that the ratio between the best and second best matching
         * distance is below some threshold. If this ratio is near 1, the match
         * is ambiguous. Good values are 0.8 for SIFT and 0.7 for SURF.
         * Set to 1.0 to disable the test.
         */
        float lowe_ratio_threshold;//最近邻的与次近邻的比例

        /**
         * Does not accept matches with distances larger than this value.
         * This needs to be tuned to the descriptor and data type used.
         * Set to FLOAT_MAX to disable the test.
         */
        float distance_threshold;//不接受距离大于此值的匹配。
        //这需要根据所使用的描述符和数据类型进行调整。
        //将其设置为FLOAT_MAX以禁用此测试。
    };

    /**
     * Feature matching result reported as two lists, each with indices in the
     * other set. An unsuccessful match is indicated with a negative index.
     */
    struct Result
    {
        /* Matches from set 1 in set 2. */
        std::vector<int> matches_1_2;
        /* Matches from set 2 in set 1. */
        std::vector<int> matches_2_1;
    };

public:
    /**
     * Matches all elements in set 1 to all elements in set 2.
     * It reports as result for each element of set 1 to which element
     * in set 2 it maches. An unsuccessful match which did not pass
     * one of the thresholds is indicated with a negative index.
     */
     //将集合1中的所有元素与集合2中的所有元素进行匹配。
     //对于集合1中的每个元素，它报告与集合2中的哪个元素匹配。
     //如果没有通过阈值的不成功匹配，则使用负索引表示。
    template <typename T>
    static void
    oneway_match (Options const& options,
        T const* set_1, int set_1_size,
        T const* set_2, int set_2_size,
        std::vector<int>* result);

    /**
     * Matches all elements in set 1 to all elements in set 2 and vice versa.
     * It reports matching results in two lists with indices.
     * Unsuccessful matches are indicated with a negative index.
     */
    //将集合1中的所有元素与集合2中的所有元素进行匹配，反之亦然。
    //它以两个带索引的列表形式报告匹配结果。
    //不成功的匹配使用负索引表示。
    template <typename T>
    static void
    twoway_match (Options const& options,
        T const* set_1, int set_1_size,
        T const* set_2, int set_2_size,
        Result* matches);


    /**
     * This function removes inconsistent matches.
     * A consistent match of a feature F1 in the first image to
     * feature F2 in the second image requires that F2 also matches to F1.
     */
     //此函数移除不一致的匹配。
     //一致的匹配指的是第一张图片中的特征 F1 匹配到第二张图片中的特征 F2，同时 F2 也要匹配到 F1。
    static void
    remove_inconsistent_matches (Result* matches);

    /**
     * Function that counts the number of valid matches.
     */
    static int
    count_consistent_matches (Result const& matches);

    /**
     * Combines matching results of different descriptors.
     */
    static void
    combine_results(Result const& sift_result,
        Result const& surf_result, Matching::Result* result);
};

/* ---------------------------------------------------------------- */

template <typename T>
void
Matching::oneway_match (Options const& options,
    T const* set_1, int set_1_size,
    T const* set_2, int set_2_size,
    std::vector<int>* result)
{
    result->clear();
    result->resize(set_1_size, -1);
    //并将所有元素初始化为-1。这里使用resize函数是为了确保result向量的大小与集合1的大小相匹配，并且初始值设置为-1是为了表示未成功匹配的情况。
    if (set_1_size == 0 || set_2_size == 0)//如果其中一个集合为空，函数将不会进行任何匹配操作，直接返回并结束函数的执行。
        return;

    // 与最近邻距离的阈值
    float const square_dist_thres = MATH_POW2(options.distance_threshold);//#define MATH_POW2(x) ((x) * (x))把阈值写成平方项

    // 以描述子为特征，计算每个特征点的最近邻和次近邻
    NearestNeighbor<T> nn;
    nn.set_elements(set_2);// 参数 T const* elements;
    // 特征点的个数
    nn.set_num_elements(set_2_size);// 参数int num_elements;
    // 设置特征描述子的维度 sift 128, surf 64

    nn.set_element_dimensions(options.descriptor_length);//参数int element_dimensions  描述子长度 182 for sift  64 for surf
    for (int i = 0; i < set_1_size; ++i)
    {
        // 每个特征点最近邻搜索的结果
        typename NearestNeighbor<T>::Result nn_result; //class NearestNeighbor{struct Result}
        // n_result 是根据模板参数 T 实例化后的 Result 类的对象。

        // feature sets 1 中第i个特征点的特征描述子
        T const* query_pointer = set_1 + i * options.descriptor_length;//options.descriptor_length 描述子长度 set_1元素地址的首位

        // 计算最近邻
        nn.find(query_pointer, &nn_result);

        // 标准1： 与最近邻的距离必须小于特定阈值
        if (nn_result.dist_1st_best > square_dist_thres)
            continue;
    //这意味着如果第一个最佳匹配的距离超过了 square_dist_thres 的值，那么程序将跳过当前的迭代，继续进行下一个迭代或循环。



        /***********************task2*************************************/
        // 标准2： 与最近邻和次紧邻的距离比必须小于特定阈值
        /*
         * 参考标准1的形式给出lowe-ratio约束
         */
        //static_cast 是 C++ 中的一个类型转换运算符，用于执行编译时的类型转换     T dist_1st_best转变为float类型的
        float square_dist_1st_best = static_cast<float>(nn_result.dist_1st_best);
        float square_dist_2st_best = static_cast<float>(nn_result.dist_2nd_best);
        float const square_lowe_thres = MATH_POW2(options.lowe_ratio_threshold);//平方一下

               /*                  */
               /*    此处添加代码    */
               /*                  */
        /*******************************10696_10015b911522757f6?bizid=10696&txSecret=63384d4bd569e29729b6995dd8a9eefb&txTime=5B93EFB6**********************************/
        //line169：这个就是最近邻比lowe_ratiod的判断
        if (static_cast<float>(nn_result.dist_1st_best) //class Matching{struct Options()}  class NearestNeighbor{struct Result()}
            / static_cast<float>(nn_result.dist_2nd_best)
            > MATH_POW2(options.lowe_ratio_threshold))
            continue;
        // 匹配成功，feature set1 中第i个特征值对应feature set2中的第index_1st_best个特征点
        result->at(i) = nn_result.index_1st_best;
        //std::vector<int>* result result->at() 是通过指针 result 访问 std::vector 对象的成员函数 at()。
        // at() 是一个用于访问向量元素的成员函数，它接受一个索引参数，并返回指定位置的元素的引用。
    }
}

template <typename T>
void
Matching::twoway_match (Options const& options,
    T const* set_1, int set_1_size,
    T const* set_2, int set_2_size,
    Result* matches)
{
    // 从feature set 2 中计算feature sets 1中每个特征点的最近邻居
    Matching::oneway_match(options, set_1, set_1_size,
        set_2, set_2_size, &matches->matches_1_2);

    // 从feature set 1 中计算feature sets 2中每个特征点的最近邻
    Matching::oneway_match(options, set_2, set_2_size,
        set_1, set_1_size, &matches->matches_2_1);
}

FEATURES_NAMESPACE_END

#endif  /* SFM_MATCHING_HEADER */

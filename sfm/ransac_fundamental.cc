/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

#include "util/system.h"
#include "math/algo.h"
#include "sfm/ransac_fundamental.h"

SFM_NAMESPACE_BEGIN

RansacFundamental::RansacFundamental (Options const& options)
    : opts(options)
{
}

void
RansacFundamental::estimate (Correspondences2D2D const& matches, Result* result)//typedef std::vector<Correspondence2D2D> Correspondences2D2D;
{
    if (this->opts.verbose_output)
    {
        std::cout << "RANSAC-F: Running for " << this->opts.max_iterations
            << " iterations, threshold " << this->opts.threshold
            << "..." << std::endl;
    }

    std::vector<int> inliers;
    inliers.reserve(matches.size());
    for (int iteration = 0; iteration < this->opts.max_iterations; ++iteration)
    {
        FundamentalMatrix fundamental;//math::Matrix<double, 3, 3> FundamentalMatrix
        this->estimate_8_point(matches, &fundamental); //八点法
        this->find_inliers(matches, fundamental, &inliers);//将内点数放入inliers里
        if (inliers.size() > result->inliers.size())//内点的个数大于最终内点的个数
            //inliers.size()放的是内点的个数，result->inliers.size()放的是ransacFundamental：：Result：：std::vector<int> ransac后inliers内点的数量
        {
            if (this->opts.verbose_output)//在控制台上输出状态信息
            {
                std::cout << "RANSAC-F: Iteration " << iteration
                    << ", inliers " << inliers.size() << " ("
                    << (100.0 * inliers.size() / matches.size())
                    << "%)" << std::endl;//它将内点数量乘以100.0，然后除以matches的数量，得到一个浮点数表示的百分比值。
            }

            result->fundamental = fundamental;//将8点法的基础矩阵f传给RansacFundamental::Result::fundamental

            std::swap(result->inliers, inliers);//result->inliers放的是ransac后inliers内点的数量，inliers放的是内点的个数 反转
            inliers.reserve(matches.size());//ransac后内点数重新预留匹配对大小的空间
        }
    }
}

void
RansacFundamental::estimate_8_point (Correspondences2D2D const& matches,
    FundamentalMatrix* fundamental)
{
    if (matches.size() < 8)
        throw std::invalid_argument("At least 8 matches required");

    /*
     * Draw 8 random numbers in the interval [0, matches.size() - 1]
     * without duplicates. This is done by keeping a set with drawn numbers.
     */
    std::set<int> result;//set容器自动排序，set容器不允许有重复的值
    while (result.size() < 8)
        result.insert(util::system::rand_int() % matches.size());//取模运算（求余运算） 取出八个随机数插入set容器

    math::Matrix<double, 3, 8> pset1, pset2;
    std::set<int>::const_iterator iter = result.begin();//只读迭代器
    for (int i = 0; i < 8; ++i, ++iter)//这里的8也就是set.size()
    {
        Correspondence2D2D const& match = matches[*iter]; //*iter解引用为int类型 写成齐次坐标形式
        pset1(0, i) = match.p1[0];//x1
        pset1(1, i) = match.p1[1];//y1
        pset1(2, i) = 1.0;
        pset2(0, i) = match.p2[0];//x2
        pset2(1, i) = match.p2[1];//y2
        pset2(2, i) = 1.0;
    }
    /* Compute fundamental matrix using normalized 8-point. */
    sfm::fundamental_8_point(pset1, pset2, fundamental);

    sfm::enforce_fundamental_constraints(fundamental);
}

void
RansacFundamental::find_inliers (Correspondences2D2D const& matches,
    FundamentalMatrix const& fundamental, std::vector<int>* result)
{
    result->resize(0);
    double const squared_thres = this->opts.threshold * this->opts.threshold;
    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        double error = sampson_distance(fundamental, matches[i]);
        //sampson_distance内点判断标准，返回值：SD = (x'Fx)^2 / ( (Fx)_1^2 + (Fx)_2^2 + (x'F)_1^2 + (x'F)_2^2 )
        if (error < squared_thres)
            result->push_back(i);
    }
}

SFM_NAMESPACE_END

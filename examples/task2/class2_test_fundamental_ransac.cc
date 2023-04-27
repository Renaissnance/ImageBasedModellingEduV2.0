/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <util/system.h>
#include <assert.h>
#include <sfm/ransac_fundamental.h>
#include "math/functions.h"
#include "sfm/fundamental.h"
#include "sfm/correspondence.h"
#include "math/matrix_svd.h"

typedef math::Matrix<double, 3, 3> FundamentalMatrix;


/**
 * \description 用于RANSAC采样成功所需要的采样次数
 * @param p -- 内点的概率
 * @param K --拟合模型需要的样本个数，对应基础矩阵num_samples=8
 * @param z  -- 预期的采样成功的概率
 *                          log(1-z)
 *       需要的采样次数 M = -----------
 *                          log(1-p^K)
 * Example: For p = 50%, z = 99%, n = 8: M = log(0.001) / log(0.99609) = 1176.
 * 需要采样1176次从而保证RANSAC的成功率不低于0.99.
 * @return
 */
int  calc_ransac_iterations (double p,
                           int K,
                           double z = 0.99){

    double prob_all_good = math::fastpow(p, K);//整个表达式的含义是将 p 的 K 次幂的结果赋值给 prob_all_good，
    double num_iterations = std::log(1.0 - z)/ std::log(1.0 - prob_all_good);
    return static_cast<int>(math::round(num_iterations));
    //math::round 是一个函数，可能是自定义实现或者某个库或框架中的函数。它接受一个参数 num_iterations，并对其进行四舍五入运算，返回最接近的整数值。
    //static_cast<int>(...)：这是一个类型转换操作，使用 static_cast 关键字将表达式的结果转换为静态 int 类型。
    // 这可能会导致 math::round(num_iterations) 返回的结果被转换为整数值，并丢弃小数部分。
}

/**
 * \description 给定基础矩阵和一对匹配点，计算匹配点的sampson 距离，用于判断匹配点是否是内点,
 * 计算公式如下：
 *              SD = (x'Fx)^2 / ( (Fx)_1^2 + (Fx)_2^2 + (x'F)_1^2 + (x'F)_2^2 )
 * @param F-- 基础矩阵
 * @param m-- 匹配对
 * @return
 */
double  calc_sampson_distance (FundamentalMatrix const& F, sfm::Correspondence2D2D const& m)
//m-- 匹配对      m.p1[0/1/2]   m.p2[0/1/2]分别表示xy1 归一化坐标
{
//SD 为 Sampson 距离，x' 为第二幅图像中的匹配点，x 为第一幅图像中的匹配点，F 为基础矩阵，
//m.p1[2]=1   m.p2[2]=1
    double p2_F_p1 = 0.0;//(x'Fx)
    p2_F_p1 += m.p2[0] * (m.p1[0] * F[0] + m.p1[1] * F[1] + F[2]);//m.p1[0]*(F*m.p2)的第一行
    p2_F_p1 += m.p2[1] * (m.p1[0] * F[3] + m.p1[1] * F[4] + F[5]);//m.p1[1]*(F*m.p2)的第二行
    p2_F_p1 +=     1.0 * (m.p1[0] * F[6] + m.p1[1] * F[7] + F[8]);//m.p1[2]*(F*m.p2)的第三行
    p2_F_p1 *= p2_F_p1;//(x'Fx)^2

    double sum = 0.0;
    sum += math::fastpow(m.p1[0] * F[0] + m.p1[1] * F[1] + F[2], 2);//(Fx)_1^2 取前两维
    sum += math::fastpow(m.p1[0] * F[3] + m.p1[1] * F[4] + F[5], 2);//(Fx)_2^2
    sum += math::fastpow(m.p2[0] * F[0] + m.p2[1] * F[3] + F[6], 2);//(x'F)_1^2 取前两维
    sum += math::fastpow(m.p2[0] * F[1] + m.p2[1] * F[4] + F[7], 2);//(x'F)_2^2

    return p2_F_p1 / sum;
}
/**
 * \description 8点发估计相机基础矩阵
 * @param pset1 -- 第一个视角的特征点
 * @param pset2 -- 第二个视角的特征点
 * @return 估计的基础矩阵
 */
 //8点法 pset1 3x8  和pset2 3x8
void calc_fundamental_8_point (math::Matrix<double, 3, 8> const& pset1
        , math::Matrix<double, 3, 8> const& pset2
        ,FundamentalMatrix &F
){
    /* direct linear transform */
    math::Matrix<double, 8, 9> A;
    for(int i=0; i<8; i++)
    {
        math::Vec3d p1  = pset1.col(i);//每一列就是一个图像坐标系的点 p1 p2 写成齐次坐标的形式
        math::Vec3d p2 = pset2.col(i);
//    p1[0]=x11 p2[0]=x12   第一个数字表示第几组   第二个数字 1表示p1  2表示p2的
//    p1[1]=y11 p2[1]=y12
        A(i, 0) = p1[0]*p2[0];
        A(i, 1) = p1[1]*p2[0];
        A(i, 2) = p2[0];
        A(i, 3) = p1[0]*p2[1];
        A(i, 4) = p1[1]*p2[1];
        A(i, 5) = p2[1];
        A(i, 6) = p1[0];
        A(i, 7) = p1[1];
        A(i, 8) = 1.0;
    }

    math::Matrix<double, 9, 9> vv;
    math::matrix_svd<double, 8, 9>(A, nullptr, nullptr, &vv);
    math::Vector<double, 9> f = vv.col(8);

    F(0,0) = f[0]; F(0,1) = f[1]; F(0,2) = f[2];
    F(1,0) = f[3]; F(1,1) = f[4]; F(1,2) = f[5];
    F(2,0) = f[6]; F(2,1) = f[7]; F(2,2) = f[8];

    /* singularity constraint */
    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd(F, &U, &S, &V);
    S(2,2)=0;
    F = U*S*V.transpose();
}

/**
 * \description 利用最小二乘法计算基础矩阵
 * @param matches--输入的匹配对 大于8对
 * @param F --基础矩阵
 */
void calc_fundamental_least_squares(sfm::Correspondences2D2D const & matches, FundamentalMatrix&F)
//matches--输入的匹配对 大于8对
{

    if (matches.size() < 8)
        throw std::invalid_argument("At least 8 points required");
    //如果 matches.size() 小于 8，那么就会抛出一个 std::invalid_argument 异常，
    // 异常的错误信息为 "At least 8 points required"，意思是 "至少需要8个点"。

    /* Create Nx9 matrix A. Each correspondence creates on row in A. */
    //其包含了两个2D点的信息，分别为p.p1和p.p2。这两个点的坐标信息被用来构建矩阵A的一行。
    std::vector<double> A(matches.size() * 9);
    //创建一个大小等于乘以 9 的A类型的向量 如果是 8点法-> 8x9=72 72维度的列向量 如果超过8对就是nx9
    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        sfm::Correspondence2D2D const& p = matches[i];
        //一对匹配点matches[0]包含了两个2D点的信息，分别为p.p1和p.p2。 i*9表示超过八对个点 每对9个数据
        A[i * 9 + 0] = p.p2[0] * p.p1[0];
        A[i * 9 + 1] = p.p2[0] * p.p1[1];
        A[i * 9 + 2] = p.p2[0] * 1.0;
        A[i * 9 + 3] = p.p2[1] * p.p1[0];
        A[i * 9 + 4] = p.p2[1] * p.p1[1];
        A[i * 9 + 5] = p.p2[1] * 1.0;
        A[i * 9 + 6] = 1.0     * p.p1[0];
        A[i * 9 + 7] = 1.0     * p.p1[1];
        A[i * 9 + 8] = 1.0     * 1.0;
    }
   // 通过循环遍历matches中的所有对应关系，矩阵A就被填充满了。每一行都对应了一个对应关系的信息，共有Nx9个元素。


    /* Compute fundamental matrix using SVD. */
    //向量的svd分解
    std::vector<double> vv(9 * 9);
    //在这里，vv被初始化为包含81个double类型的元素的向量，可以用于存储9x9矩阵的数据。
    // S U V V的矩阵大小一定的因为A：MXN U：MXN S：NXN U ：NXN 在行数未知的时候 只有S V能确定下来
    math::matrix_svd<double>(&A[0], matches.size(), 9, nullptr, nullptr, &vv[0]);
//template<T>
// void matrix_svd(const T *mat_a, int rows（行）, int cols（列）, T *mat_u, T *vec_s, T *mat_v, const T ε = T(1e-12))
//T: 模板参数，表示矩阵元素的数据类型，可以是 float、double 等。
//mat_a: 输入参数，指向输入矩阵的指针，表示待进行奇异值分解的矩阵。
//mat_a 是一个指向矩阵 A 第一个元素的指针。通过 &A[0] 可以获取 A 数组的第一个元素的地址，从而得到 mat_a 的值。
//在 C++ 中，数组名本质上是一个指向数组第一个元素的指针，因此可以使用 &A[0] 或者简写为 A 来表示指向数组 A 第一个元素的指针。
// 在函数调用时，将 mat_a 设置为 &A[0]，即传递了数组 A 的首地址给 matrix_svd 函数，从而使得函数能够访问和处理数组 A 中的数据。
// 注意，在使用指针访问数组元素时，需要确保数组的内存布局是连续的，并且访问的索引不超过数组的范围，以避免越界访问和未定义行为。
//rows: 输入参数，表示输入矩阵的行数。
//cols: 输入参数，表示输入矩阵的列数。
//mat_u: 输出参数，指向保存矩阵 U 的指针，U 是由左奇异向量组成的正交矩阵。
//vec_s: 输出参数，指向保存奇异值的指针，奇异值是一个向量，保存了矩阵 A 的奇异值。
//mat_v: 输出参数，指向保存矩阵 V 的指针，V 是由右奇异向量组成的正交矩阵。
//ε: 输入参数，表示用于控制精度的一个小数，默认值为 1e-12。

    /* Use last column of V as solution. */ //使用V的最后一列作为解 vv是9x9 0-8 给基础矩阵F赋值
    for (int i = 0; i < 9; ++i)
        F[i] = vv[i * 9 + 8];

    /* singularity constraint */
    //对F进行svd分解 奇异值约束
    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd(F, &U, &S, &V);
    S(2,2)=0;
    F = U*S*V.transpose();
}
/**
 * \description 给定匹配对和基础矩阵，计算内点的个数
 * @param matches 匹配对
 * @param F 基础矩阵
 * @return
 */
std::vector<int> find_inliers(sfm::Correspondences2D2D const & matches,FundamentalMatrix const & F,const double & thresh)
//用于从给定的 2D-2D 匹配对 matches 中找到符合阈值条件的内点（inliers）。
//matches：类型为 sfm::Correspondences2D2D 的匹配对，表示两幅图像间的2D点对应关系。
//F：类型为 FundamentalMatrix 的基础矩阵，用于计算 Sampson 距离。
//thresh：类型为 double 的阈值，用于判断点对是否为内点的阈值。
    {
    const double squared_thresh = thresh* thresh;

    std::vector<int> inliers;  //内点 存的是索引的值
    for(int i=0; i< matches.size(); i++){
        double error = calc_sampson_distance(F, matches[i]);
        //m是一对 m.p1 m.p2  matches[i]表示有i个匹配对 匹配点的sampson 距离，用于判断匹配点是否是内点,m-- 匹配对
        if(error< squared_thresh){
            inliers.push_back(i);
        }
    }
    return inliers;
}
//将阈值 thresh 的平方存储在 squared_thresh 中，避免在后续比较中进行平方运算。
//初始化一个空的 std::vector<int> 类型的内点索引向量 inliers，用于存储符合阈值条件的内点的索引。
//遍历 matches 中的每一个匹配对，使用函数 calc_sampson_distance 计算该匹配对的 Sampson 距离，保存在变量 error 中。
//判断 error 是否小于 squared_thresh，如果是，则将该匹配对的索引 i 添加到 inliers 中。
//循环结束后，返回存储了符合阈值条件的内点索引的 inliers 向量。


int main(int argc, char *argv[]){

    /** 加载归一化后的匹配对 */
    sfm::Correspondences2D2D corr_all;//匹配对matches
//这段代码是用于加载从文件中读取的归一化后的匹配对数据，
// 并将其存储到一个名为corr_all的sfm::Correspondences2D2D类型的变量中。
    std::ifstream in("/home/ros/ImageBasedModellingEduV1.0/examples/task2/correspondences.txt");
    assert(in.is_open());
//代码首先通过std::ifstream类创建一个文件输入流in，并指定文件路径为"./examples/task2/correspondences.txt"。
// 接着，通过assert语句检查文件是否成功打开，如果未成功打开，会触发断言错误。

    std::string line, word;
    int n_line = 0;
    while(getline(in, line))
    {
//接下来，代码通过std::getline()函数逐行读取文件内容。每一行的内容会存储在名为line的字符串变量中。
// 在每一行中，代码通过std::stringstream类创建一个字符串流stream，以便对每一行的内容进行解析。

        std::stringstream stream(line);
        if(n_line==0)
        {
            int n_corrs = 0;
            stream>> n_corrs;
            corr_all.resize(n_corrs);
//在解析过程中，代码首先判断n_line的值。当n_line为0时，表示读取到的是第一行，其中包含了匹配对的数量信息。
// 代码通过stream从第一行中读取匹配对的数量，并将其存储到n_corrs变量中。然后，通过调用corr_all.resize(n_corrs)，
// 为corr_all变量的容器分配足够的空间，以存储接下来读取到的匹配对数据。
            n_line ++;
            continue;
        }
        if(n_line>0)
        {
//当n_line大于0时，表示读取到的是匹配对数据的行。代码通过stream从当前行中依次读取四个浮点数值，
// 并将其分别存储到corr_all变量中的对应成员变量中，
// 包括p1[0]、p1[1]、p2[0]和p2[1]，分别表示第一幅图像中匹配点的x和y坐标，以及第二幅图像中匹配点的x和y坐标。
            stream>>corr_all[n_line-1].p1[0]>>corr_all[n_line-1].p1[1]
                  >>corr_all[n_line-1].p2[0]>>corr_all[n_line-1].p2[1];
        }
        n_line++;
//最后，n_line会自增，以继续读取下一行的内容，直至文件读取完毕，完成匹配对数据的加载过程。
    }

    /* 计算采用次数 */
    const float inlier_ratio =0.5;
    const int n_samples=8;
    int n_iterations = calc_ransac_iterations(inlier_ratio, n_samples); // 用于RANSAC采样成功所需要的采样次数 z已经给初值了

    // 用于判读匹配对是否为内点
    const double inlier_thresh = 0.0015; //thresh：类型为 double 的阈值，用于判断点对是否为内点的阈值。

    // ransac 最终估计的内点
    std::vector<int> best_inliers;  //  std::vector<int> inliers;内点 存的是索引的值

    std::cout << "RANSAC-F: Running for " << n_iterations
              << " iterations, threshold " << inlier_thresh
              << "..." << std::endl;
    for(int i=0; i<n_iterations; i++)
    {

        /* 1.0 随机找到8对不重复的匹配点 */
        std::set<int> indices;
        while(indices.size()<8)
        {
            indices.insert(util::system::rand_int() % corr_all.size());
    //util::system::rand_int() 是一个函数调用，用于生成一个随机整数。util::system::rand_int() 函数可能
    // 属于 util 命名空间下的 system 子命名空间，用于操作系统相关的功能，例如生成随机数。
    // util::system::rand_int() 函数生成的随机整数除以 corr_all.size() 取余数。
    // 然后，将这个索引插入到名为indices的std::set容器中，这样可以确保选出的索引不重复。
    // 这个过程会循环执行，直到indices中有8个元素，即随机选择了8对匹配点。
        }

        math::Matrix<double, 3, 8> pset1, pset2;
        //从一个名为corr_all的匹配点对中随机选择8对不重复的点，一个xy坐标 齐次坐标形式 并将其分别存储在两个3x8的矩阵pset1和pset2中。
        std::set<int>::const_iterator iter = indices.cbegin();//常量迭代器 iter，指向 indices 容器中第一个元素的位置，但不允许修改容器中的元素。
        for(int j=0; j<8; j++, iter++)
        {
            sfm::Correspondence2D2D const & match = corr_all[*iter];//*iter解应用为indces存储的数值 索引  match/corr_all都是匹配对
            pset1(0, j) = match.p1[0];
            pset1(1, j) = match.p1[1];
            pset1(2, j) = 1.0;

            pset2(0, j) = match.p2[0];
            pset2(1, j) = match.p2[1];
            pset2(2, j) = 1.0;
        }

        /*2.0 8点法估计相机基础矩阵*/
        FundamentalMatrix F;
        calc_fundamental_8_point(pset1, pset2,F);

        /*3.0 统计所有的内点个数*/
        std::vector<int> inlier_indices = find_inliers(corr_all, F, inlier_thresh);

        if(inlier_indices.size()> best_inliers.size())//是用来判断当前迭代得到的内点数量是否大于之前的最优内点数量，以确定是否更新最优内点。
        {

//            std::cout << "RANSAC-F: Iteration " << i
//                      << ", inliers " << inlier_indices.size() << " ("
//                      << (100.0 * inlier_indices.size() / corr_all.size())
//                      << "%)" << std::endl;
            best_inliers.swap(inlier_indices);//先计算内点个数 ，最优内点数还未赋值 等内点数算了一个初值再迭代求最优
//如果当前迭代得到的内点数量比之前的最优内点数量大， 将当前迭代的内点索引赋值给 best_inliers，（swap）更新最优内点。
        }
    }

    sfm::Correspondences2D2D corr_f;
    for(int i=0; i< best_inliers.size(); i++)
    {
        corr_f.push_back(corr_all[best_inliers[i]]);
//corr_all 274组数据 best_inliers[i]表示 符合内点要求索引值    内点存的是索引的值 corr_all[]取出索引值对应的行
//corr_all[best_inliers[i]] 表示根据 best_inliers 中的当前索引 i，从 corr_all 中获取对应的匹配对。
    }
//从 corr_all 中提取出 RANSAC 算法得到的最优内点匹配对，并将其存储在 corr_f 中，以便后续使用。

    /*利用所有的内点进行最小二乘估计*/
    FundamentalMatrix F;
    calc_fundamental_least_squares(corr_f, F);

    std::cout<<"inlier number: "<< best_inliers.size()<<std::endl;
    std::cout<<"F\n: "<< F<<std::endl;

//    std::cout<<"result should be: \n"
//             <<"inliner number: 272\n"
//             <<"F: \n"
//             <<"-0.00961384 -0.0309071 0.703297\n"
//             <<"0.0448265 -0.00158655 -0.0555796\n"
//             <<"-0.703477 0.0648517 -0.0117791\n";

    return 0;
}
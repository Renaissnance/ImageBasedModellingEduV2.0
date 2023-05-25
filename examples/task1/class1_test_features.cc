/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>

#include "util/file_system.h"
#include "util/timer.h"
#include "core/image.h"
#include "core/image_tools.h"
#include "core/image_io.h"

#include "features/surf.h"
#include "features/sift.h"
#include "visualizer.h"

bool
sift_compare (features::Sift::Descriptor const& d1, features::Sift::Descriptor const& d2)
{
    return d1.scale > d2.scale;
}

int
main (int argc, char** argv)
//argc 是 argument count的缩写，表示传入main函数的参数个数；
//argv 是 argument vector的缩写，表示传入main函数的参数序列或指针，并且第一个参数argv[0]一定是程序的名称，
//并且包含了程序所在的完整路径，所以确切的说需要我们输入的main函数的参数个数应该是argc-1个；
//在没有参数传入的情况下，保存程序名称的第一个变量argv[0]依然存在。
{
    if (argc < 3)//在这段代码中，需要传入两个参数：一个是图像的路径，另一个是输出文件的路径，没有这两个参数程序就无法运行。
        // 因此，如果命令行参数数量小于3，程序会输出错误信息并返回1表示程序异常退出。
    {
        std::cerr << "Syntax: " << argv[0] << " <image>" << "out put file name path without .png format"<<std::endl;
        return 1;
    }

    /* 加载图像*/
    core::ByteImage::Ptr image;
    std::string image_filename = argv[1];
    try
    {
        std::cout << "Loading " << image_filename << "..." << std::endl;
        image = core::image::load_file(image_filename);
        //image = core::image::rescale_half_size<uint8_t>(image);
        //image = core::image::rescale_half_size<uint8_t>(image);
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    //这段代码是用来捕获可能发生的异常并处理异常的代码块。在 C++ 编程中，如果代码发生了错误或异常，程序将停止运行并抛出一个异常对象，该对象可以包含错误信息和其他相关信息。
    // 在这个代码块中，我们捕获了 std::exception 类型的异常，这是一个通用的 C++ 异常类，它可以表示各种类型的异常。如果程序抛出了 std::exception 类型的异常，
    // 我们会将其错误信息打印到标准错误输出流 std::cerr 中，并返回一个非零的错误代码（1），以通知调用方发生了错误。


    /* SIFT 特征检测. */
    features::Sift::Descriptors sift_descr;
    features::Sift::Keypoints sift_keypoints;
    {
        features::Sift::Options sift_options;
        sift_options.verbose_output = true;
        sift_options.debug_output = true;
        features::Sift sift(sift_options);
        sift.set_image(image);

        util::WallTimer timer;//util::WallTimer 是一个计时器类，它可以用来测量程序的运行时间。被用来测量 SIFT 特征检测的运行时间。
        sift.process(); // 主程序
        std::cout << "Computed SIFT features in "
                  << timer.get_elapsed() << "ms." << std::endl;

        sift_descr = sift.get_descriptors();
        sift_keypoints = sift.get_keypoints();
    }

    // 对特征点按照尺度进行排序
    std::sort(sift_descr.begin(), sift_descr.end(), sift_compare);
    //first 和 last 分别指向待排序范围的起始和终止位置（不包括终止位置）comp用于定义排序规则。如果不提供 comp，则使用默认的 < 运算符进行比较。

    std::vector<features::Visualizer::Keypoint> sift_drawing;
    for (std::size_t i = 0; i < sift_descr.size(); ++i)
    {
        features::Visualizer::Keypoint kp;
        kp.orientation = sift_descr[i].orientation;
        kp.radius = sift_descr[i].scale;
        kp.x = sift_descr[i].x;
        kp.y = sift_descr[i].y;
        sift_drawing.push_back(kp);
    }

    core::ByteImage::Ptr sift_image = features::Visualizer::draw_keypoints(image,
        sift_drawing, features::Visualizer::RADIUS_BOX_ORIENTATION);

    /* 保存图像文件名 */
    std::string filename =argv[2];
    std::string sift_out_fname = filename + util::fs::replace_extension
        (util::fs::basename(image_filename), "sift.png");
    std::cout << "保存图像: " << sift_out_fname << std::endl;
    core::image::save_file(sift_image, sift_out_fname);

    return 0;
}

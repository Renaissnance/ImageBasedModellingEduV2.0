/*
 * Copyright (C) 2015, Benjamin Richter, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_ALIGNED_MEMORY_HEADER
#define UTIL_ALIGNED_MEMORY_HEADER

#include <cstdint>
#include <vector>

#include "util/defines.h"
#include "util/aligned_allocator.h"

UTIL_NAMESPACE_BEGIN

template <typename T, size_t ALIGNMENT = 16>
using AlignedMemory = std::vector<T, AlignedAllocator<T, ALIGNMENT>>;
//具体而言，AlignedMemory 是一个使用模板类 std::vector 实例化得到的类型，该模板类的第一个模板参数是 T，表示容器中存储的元素类型。
// 第二个模板参数是 AlignedAllocator<T, ALIGNMENT>，表示用于分配内存的分配器类型，其中 T 是元素类型，ALIGNMENT 是对齐方式。
//AlignedAllocator 可能是一个自定义的分配器类，用于在内存分配时进行特定的对齐操作。通过指定特定的对齐方式，可以确保分配的内存按照指定的对齐边界对齐，以满足特定的内存对齐需求。

UTIL_NAMESPACE_END

#endif /* UTIL_ALIGNED_MEMORY_HEADER */

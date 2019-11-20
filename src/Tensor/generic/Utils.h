#pragma once
#include "Exception.h"
#include "General.h"
#include <vector>
/*
添加了常用的函数
例如将形状转换成大小和角标偏移
*/
namespace zf {
	//将形状转换成大小，不提供检查机制
	Size shape_to_size(const Shape& shape);
	//将形状转换成角标偏移，不提供检查机制
	Stride shape_to_stride(const Shape& shape);
	//自动设置形状，将-1替换成其他值，不提供检查机制
	void auto_set_shape(Shape& shape, Size target_size);
}
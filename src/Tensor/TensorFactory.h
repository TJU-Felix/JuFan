#pragma once
#include "generic/General.h"
#include "generic/Exception.h"
#include "generic/Fill.h"
#include "TensorImpl.h"
#include <memory>
/*
创建张量
*/
namespace zf {
	//创建常数张量
	Tensor constant(const Shape& shape, DataType value);
	Tensor zeros(const Shape& shape);
	Tensor ones(const Shape& shape);
	//创建随机张量
	Tensor rand(const Shape& shape, double mean=0, double std=1);
	//创建数
	Tensor scalar(DataType value);
}//zf

#pragma once
#include "generic/General.h"
#include "generic/Exception.h"
#include "generic/Fill.h"
#include "TensorImpl.h"
#include <memory>
/*
��������
*/
namespace zf {
	//������������
	Tensor constant(const Shape& shape, DataType value);
	Tensor zeros(const Shape& shape);
	Tensor ones(const Shape& shape);
	//�����������
	Tensor rand(const Shape& shape, double mean=0, double std=1);
	//������
	Tensor scalar(DataType value);
}//zf

#pragma once
#include "TensorImpl.h"
/*
提供了张量运算操作
*/
namespace zf {
	Tensor operator+(const Tensor& left, const Tensor& right);
	Tensor operator+(const Tensor& left, const DataType& _data);
	Tensor operator+(const DataType& _data, const Tensor& rhs);
	Tensor operator-(const Tensor& left, const Tensor& right);
	Tensor operator-(const Tensor& left, const DataType& _data);
	Tensor operator-(const DataType& _data, const Tensor& rhs);
	Tensor operator*(const Tensor& left, const Tensor& right);
	Tensor operator*(const Tensor& left, const DataType& _data);
	Tensor operator*(const DataType& _data, const Tensor& rhs);
	Tensor operator/(const Tensor& left, const Tensor& right);
	Tensor operator/(const Tensor& left, const DataType& _data);
	Tensor operator/(const DataType& _data, const Tensor& rhs);
	Tensor& operator+=(Tensor& left, const Tensor& right);
	Tensor& operator+=(Tensor& left, const DataType& right);
	Tensor& operator-=(Tensor& left, const Tensor& right);
	Tensor& operator-=(Tensor& left, const DataType& right);
	Tensor& operator*=(Tensor& left, const Tensor& right);
	Tensor& operator*=(Tensor& left, const DataType& right);
	Tensor& operator/=(Tensor& left, const Tensor& right);
	Tensor& operator/=(Tensor & left, const DataType& right);
	Tensor pow(Tensor& left, double e);
	Tensor& pow_(Tensor& left, double e);
}
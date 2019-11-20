#pragma once
#include "TensorAutograd.h"
#include "operator/OperatorHandle.h"
/*
定义了张量的各种运算
包括加减乘除幂、最大最小、拷贝索引
*/
namespace ai {

	Tensor operator+(const Tensor& left, const Tensor& right);
	Tensor operator+(const DataType& left, const Tensor& right);
	Tensor operator+(const Tensor& left, const DataType& right);
	Tensor operator-(const Tensor& left, const Tensor& right);
	Tensor operator-(const DataType& left, const Tensor& right);
	Tensor operator-(const Tensor& left, const DataType& right);
	Tensor operator*(const Tensor& left, const Tensor& right);
	Tensor operator*(const DataType& left, const Tensor& right);
	Tensor operator*(const Tensor& left, const DataType& right);
	Tensor operator/(const Tensor& left, const Tensor& right);
	Tensor operator/(const DataType& left, const Tensor& right);
	Tensor operator/(const Tensor& left, const DataType& right);

	/*******************************************************
张量运算。
根据volatile分为两类：
一、若volatile为true
则张量计算出的中间变量将不进行hold。
二、false
分为三类
1. 双张量运算。参与者有两个张量。
	参与张量用shared_ptr维持计算结果，同时计算结果也用shared_ptr指向参与张量。
	这里是考虑到了临时变量的情况，以及为了便于释放中间变量。
	由于这里会产生循环引用，因此需要手动释放。
2. 多张量运算

3. 单张量运算

*******************************************************/




}


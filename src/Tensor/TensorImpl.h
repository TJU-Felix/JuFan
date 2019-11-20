#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <math.h>
#include "TensorStorage.h"
#include "generic/Exception.h"
/*
张量
*/
namespace zf {
	class Tensor
	{
		//typedef std::vector<Int64> Shape;
	protected:
		Shape _stride;			//角标偏移 2
		Shape _shape;			//形状 4
		TensorStorage _storage;	//底层存储 5

		int _auto_set_by_shape();//自动根据形状设置其他参数
	public:
		/********************************************
		构造函数：*
		创建张量请用工厂函数
		引用语义，如果想要拷贝请调用copy函数
		*********************************************/
		Tensor() = default;
		Tensor(Tensor&& t) = default;
		Tensor(const Tensor& t) = default;
		Tensor(const TensorStorage& storage, Shape shape);
		//释放成空张量
		void release();
		/********************************************
		拷贝操作：**
		赋值符号均为引用语义
		拷贝请用copy函数
		*********************************************/
		Tensor& operator=(const Tensor& t) = default;
		Tensor copy() const;
		int copy_(const Tensor& rhs);
		/********************************************
		只读，访问属性：*
		*********************************************/
		const Shape& shape() const;
		Size size() const;
		void display_information() const;//TODO
		void display() const;
		TensorStorage storage_unsafe() const;
		bool isNULL() const;//是否是空张量
		bool isScalar() const;//是否是张量
		bool isVector() const;//是否是一维向量
		bool isTensor() const;//是否是多维张量
		bool isMatrix() const;//是否是二维矩阵
		/********************************************
		修改，不影响底层存储，只影响观察模式：**
		*********************************************/
		Tensor operator=(DataType value);
		Tensor reshape(const Shape& shape) const;
		Tensor& reshape_(const Shape& shape);
		/********************************************
		索引和切片，既影响底层存储，又影响观察模式：****
		*********************************************/
		Tensor operator[](Rank rank) const;
		Tensor slice(Rank dim, Int64 begin, Int64 end, Int64 stride = 1) const;//TODO

		/*
		强制转换
		*/
		operator double() const;
		operator Int64() const;
		//定义负号
		Tensor operator-() const;
		//
		Tensor sum() const;
		Tensor max() const;
		Tensor min() const;

		/********************************************
		运算：*****
		TensorFunctions.h提供了更全面的运算
		*********************************************/

	};
}


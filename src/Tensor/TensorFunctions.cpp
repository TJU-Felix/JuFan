#include "TensorFunctions.h"
#include "TensorFactory.h"
#define TENSOR_OP_VALUE(op) auto _size = left.size();\
							auto _s = left.storage_unsafe();\
						for(int i=0;i<_size;i++)\
						_s[i]##op##right;

#define TENSOR_OP_(op) auto _s_1 = left.storage_unsafe(), _s_2=right.storage_unsafe()\
						;auto _size=left.size();\
						for(int i=0;i<_size;i++)\
						_s_1[i]##op##_s_2[i];
namespace zf {
	Tensor operator+(const Tensor& left, const Tensor& right)
	{
		Tensor result = left.copy();
		return result += right;
	}
	Tensor operator+(const Tensor& left, const DataType& _data)
	{
		Tensor result = left.copy();
		return result += _data;
	}
	Tensor operator+(const DataType& _data, const Tensor& right)
	{
		Tensor result = right.copy();
		return result += _data;
	}
	Tensor operator-(const Tensor& left, const Tensor& right)
	{
		Tensor result = left.copy();
		return result -= right;
	}
	Tensor operator-(const Tensor& left, const DataType& _data)
	{
		Tensor result = left.copy();
		return result -= _data;
	}
	Tensor operator-(const DataType& _data, const Tensor& right)
	{
		Tensor temp = constant(right.shape(), _data);
		return temp -= right;
	}
	Tensor operator*(const Tensor& left, const Tensor& right)
	{
		Tensor result = left.copy();
		return result *= right;
	}
	Tensor operator*(const Tensor& left, const DataType& _data)
	{
		Tensor result = left.copy();
		return result *= _data;
	}
	Tensor operator*(const DataType& _data, const Tensor& right)
	{
		Tensor result = right.copy();
		return result *= _data;
	}
	Tensor operator/(const Tensor& left, const Tensor& right)
	{
		Tensor result = left.copy();
		return result /= right;
	}
	Tensor operator/(const Tensor& left, const DataType& _data)
	{
		Tensor result = left.copy();
		return result /= _data;
	}
	Tensor operator/(const DataType& _data, const Tensor& right)
	{
		Tensor temp = constant(right.shape(), _data);
		return temp /= right;
	}

	Tensor pow(Tensor& left, double e)
	{
		Tensor result = left.copy();
		return pow_(result, e);
	}
	Tensor& pow_(Tensor& left, double e)
	{
		auto _size = left.size();
		for (int i = 0; i < _size; i++)
			left.storage_unsafe()[i] = std::pow(left.storage_unsafe()[i], e);
		return left;
	}

	Tensor& operator+=(Tensor& left, const Tensor& right)
	{
		//TENSOR_CHECK_SAME_SHAPE(_shape, right._shape);
		TENSOR_OP_(+=);
		return left;
	}
	Tensor& operator+=(Tensor& left, const DataType& right)
	{
		TENSOR_OP_VALUE(+=);
		return left;
	}
	Tensor& operator-=(Tensor& left, const Tensor& right)
	{
		//TENSOR_CHECK_SAME_SHAPE(_shape, right._shape);//检查形状
		TENSOR_OP_(-=);
		return left;
	}
	Tensor& operator-=(Tensor& left, const DataType& right)
	{
		TENSOR_OP_VALUE(-=);
		return left;
	}
	Tensor& operator*=(Tensor& left, const Tensor& right)
	{
		//TENSOR_CHECK_SAME_SHAPE(_shape, right._shape);//检查形状
		TENSOR_OP_(*=);
		return left;
	}
	Tensor& operator*=(Tensor& left, const DataType& right)
	{
		TENSOR_OP_VALUE(*=);
		return left;
	}
	Tensor& operator/=(Tensor& left, const Tensor& right)
	{
		//TENSOR_CHECK_SAME_SHAPE(_shape, right._shape);//检查形状
		TENSOR_OP_(/=);
		return left;
	}
	Tensor& operator/=(Tensor& left, const DataType& right)
	{
		TENSOR_OP_VALUE(/=);
		return left;
	}

}
#undef TENSOR_OP_

#undef TENSOR_OP_VALUE
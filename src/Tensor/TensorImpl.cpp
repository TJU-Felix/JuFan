#include "TensorImpl.h"
#include "TensorFactory.h"
#include "generic/Utils.h"
#include <iostream>
namespace zf {

	Tensor::Tensor(const TensorStorage& storage, Shape shape)
	{
		_storage = storage;
		_shape = shape;
		_auto_set_by_shape();
	}
	void Tensor::release()
	{
		_shape.clear();
		_stride.clear();
		_storage.release();
	}
	Tensor Tensor::copy() const
	{
		Tensor temp;
		temp.copy_(*this);
		return temp;
	}
	int Tensor::copy_(const Tensor& rhs)//从其他的张量拷贝数据，注意维度相等
	{
		_storage=rhs._storage.copy();//拷贝底层
		_stride = rhs._stride;
		_shape = rhs._shape;
		return 0;
	}
	Tensor Tensor::operator[](Rank rank) const
	{
		Tensor temp(*this);
		temp._storage = _storage.slice(rank*_stride[0], 0, 1, _stride[0], 1, 1);
		temp._shape.erase(temp._shape.begin());//删除第一个维度
		temp._auto_set_by_shape();
		return temp;
	}
	Tensor Tensor::slice(Rank dim, Int64 begin, Int64 end, Int64 stride) const//切片
	{
		Tensor temp(*this);
		size_t n_segment, n_per_segment, base, stride_between_segment, stride_in_segment, persize;
		base = begin * _stride[dim];//偏移量
		persize = _stride[dim];
		n_per_segment = (end - begin) / stride;//每个段的碎片大小
		if ((end - begin) % stride)
			n_per_segment++;
		n_segment = 1;//段的数量
		for (int i = 0; i < dim; i++)
			n_segment *= _shape[i];
		stride_between_segment = 0;
		if (dim)
			stride_between_segment = _stride[dim - 1];
		stride_in_segment = _stride[dim] * stride;
		//设置底层存储
		temp._storage = _storage.slice(base,
			stride_between_segment,
			stride_in_segment,
			persize,
			n_per_segment,
			n_segment);
		//设置形状
		temp._shape.clear();
		for (int i = 0; i < dim ; i++)
			temp._shape.push_back(_shape[i]);
		temp._shape.push_back(n_per_segment);
		temp._auto_set_by_shape();
		return temp;
	}
	//只读操作
	const Shape& Tensor::shape() const
	{
		return _shape;
	}
	Size Tensor::size() const
	{
		return _storage.size();
	}
	bool Tensor::isNULL() const//是否是张量
	{
		return size() == 0;
	}
	bool Tensor::isScalar() const//是否是张量
	{
		return _shape.size() == 0;
	}
	bool Tensor::isVector() const//是否是一维向量
	{
		return _shape.size() == 1;
	}
	bool Tensor::isTensor() const//是否是多维张量
	{
		return _shape.size() >= 3;
	}
	bool Tensor::isMatrix() const//是否是二维矩阵
	{
		return _shape.size() == 2;
	}
	//赋值操作
	Tensor Tensor::operator=(DataType value)
	{
		_storage = value;
		return *this;
	}

	void Tensor::display_information() const
	{
		using std::cout;
		using std::endl;
		if (isNULL()) {
			cout << "NULL Tensor" << endl;
			return;
		}
		cout << "CPU Double {";
		for (auto i : _shape)
			cout << i << " ";
		cout << "} ";
		cout << "<" << _storage.bottom_dataptr_unsafe() << ">" << endl;
	}

	void Tensor::display() const
	{
		using std::cout;
		using std::endl;
		display_information();
		if (isNULL())
			return;
		if (isScalar())
		{
			cout << storage_unsafe()[0] << endl;
		}
		else if (isVector())
		{
			cout << "[";
			for (int i = 0; i < _shape[0]; i++)
				cout << DataType((*this)[i]) << " " << endl;
			cout << "]" << endl;
		}
		else if (isMatrix())
		{
			cout << "[";
			for (int i = 0; i < _shape[0]; i++) {
				for (int j = 0; j < _shape[1]; j++) {
					cout << DataType((*this)[i][j]) << " ";
				};
				if (i < _shape[0] - 1)
					cout << endl;
				else
					cout << "]" << endl;
			}
		}
		else {//isTensor
			cout << "is a Tensor" << endl;
		}
	}
	Tensor Tensor::reshape(const Shape& shape) const
	{
		//TENSOR_CHECK_SHAPE_INIT(shape);
		//TENSOR_CHECK_SAME_SIZE(_shape, shape);
		auto result = (*this).copy();
		result._shape = shape;
		result._auto_set_by_shape();
		return result;
	}
	Tensor& Tensor::reshape_(const Shape& shape)
	{
		//TENSOR_CHECK_SHAPE_INIT(shape);
		//TENSOR_CHECK_SAME_SIZE(_shape, shape);
		_shape = shape;
		_auto_set_by_shape();
		return *this;
	}
	int Tensor::_auto_set_by_shape()
	{
		_stride = shape_to_stride(_shape);
		return 0;
	}
	TensorStorage Tensor::storage_unsafe() const
	{
		return _storage;
	}
	Tensor::operator double() const
	{
		if (isScalar())
			return _storage[0];
		else
			return NULL;
	}
	Tensor::operator Int64() const
	{
		if (isScalar())
			return Int64(_storage[0]);
		else
			return NULL;
	}
	Tensor Tensor::operator-() const
	{
		auto a = this->copy();
		DataType*p = a._storage.bottom_dataptr_unsafe();
		for (int i = 0; i < a._storage.bottom_size(); i++) {
			*p = -(*p);
		}
		return a;
	}
	Tensor Tensor::sum() const
	{
		DataType s = 0;
		for (int i = 0; i < _storage.size(); i++)
			s += _storage[i];
		return scalar(s);
	}
	Tensor Tensor::max() const
	{
		DataType s = _storage[0];
		for (int i = 1; i < _storage.size(); i++)
			if (s < _storage[i])
				s = _storage[i];
		return scalar(s);
	}
	Tensor Tensor::min() const
	{
		DataType s = _storage[0];
		for (int i = 1; i < _storage.size(); i++)
			if (s > _storage[i])
				s = _storage[i];
		return scalar(s);
	}
}
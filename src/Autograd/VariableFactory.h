#pragma once
namespace ai {
	/*
	创建各种张量
	*/
	inline Tensor constant(const Shape& shape, DataType value)
	{
		std::shared_ptr<_Variable_> impl = std::make_shared<_Variable_>(zf::constant(shape, value));
		return Tensor(impl, false);
	}
	inline Tensor zero(const Shape& shape, DataType value)
	{
		return constant(shape, 0);
	}
	inline Tensor ones(const Shape& shape, DataType value)
	{
		return constant(shape, 1);
	}
	inline Tensor rand(const Shape& shape, double mean = 0, double std = 1)
	{
		std::shared_ptr<_Variable_> impl = std::make_shared<_Variable_>(zf::rand(shape, mean, std));
		return Tensor(impl, false);
	}
	inline Tensor scalar(DataType value)
	{
		std::shared_ptr<_Variable_> impl = std::make_shared<_Variable_>(zf::scalar(value));
		return Tensor(impl, false);
	}

}
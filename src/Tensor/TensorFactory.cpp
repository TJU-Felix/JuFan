#include "TensorFactory.h"
#include "generic/Utils.h"
#include "generic/Fill.h"
//#include "TensorImpl.h"
namespace zf {
	Tensor constant(const Shape& shape, DataType value)
	{
		TensorStorage store(shape_to_size(shape));
		fill_constant(store.bottom_dataptr_unsafe(), store.bottom_size(), value);
		return Tensor(store, shape);
	}
	Tensor zeros(const Shape& shape)
	{
		return constant(shape, 0);
	}
	Tensor ones(const Shape& shape)
	{
		return constant(shape, 1);
	}
	Tensor rand(const Shape& shape, double mean, double std)
	{
		TensorStorage store(shape_to_size(shape));
		fill_randnormal(store.bottom_dataptr_unsafe(), store.bottom_size(), mean, std);
		return Tensor(store, shape);
	}
	Tensor scalar(DataType value)
	{
		TensorStorage store(1);
		*(store.bottom_dataptr_unsafe()) = value;
		return Tensor(store, {});
	}
}
#include"Utils.h"
namespace zf {
	Size shape_to_size(const Shape& shape)
	{
		Size size = 1;
		for (auto i : shape)
			size *= i;
		return size;
	}
	//将形状转换成角标偏移，不提供检查机制
	Stride shape_to_stride(const Shape& shape)
	{
		size_t s = shape.size();
		Stride stride(s);
		Rank temp = 1;
		for (int i = int(s) - 1; i >= 0; i--) {
			stride[i] = temp;
			temp *= shape[i];
		}
		return stride;
	}
	//自动设置形状，将-1替换成其他值，不提供检查机制
	void auto_set_shape(Shape& shape, Size target_size)
	{

	}
}
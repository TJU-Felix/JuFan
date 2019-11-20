#include"Utils.h"
namespace zf {
	Size shape_to_size(const Shape& shape)
	{
		Size size = 1;
		for (auto i : shape)
			size *= i;
		return size;
	}
	//����״ת���ɽǱ�ƫ�ƣ����ṩ������
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
	//�Զ�������״����-1�滻������ֵ�����ṩ������
	void auto_set_shape(Shape& shape, Size target_size)
	{

	}
}
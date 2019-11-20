#pragma once
#include "Exception.h"
#include "General.h"
#include <vector>
/*
����˳��õĺ���
���罫��״ת���ɴ�С�ͽǱ�ƫ��
*/
namespace zf {
	//����״ת���ɴ�С�����ṩ������
	Size shape_to_size(const Shape& shape);
	//����״ת���ɽǱ�ƫ�ƣ����ṩ������
	Stride shape_to_stride(const Shape& shape);
	//�Զ�������״����-1�滻������ֵ�����ṩ������
	void auto_set_shape(Shape& shape, Size target_size);
}
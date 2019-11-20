#pragma once
#include "TensorAutograd.h"
#include "operator/OperatorHandle.h"
/*
�����������ĸ�������
�����Ӽ��˳��ݡ������С����������
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
�������㡣
����volatile��Ϊ���ࣺ
һ����volatileΪtrue
��������������м������������hold��
����false
��Ϊ����
1. ˫�������㡣������������������
	����������shared_ptrά�ּ�������ͬʱ������Ҳ��shared_ptrָ�����������
	�����ǿ��ǵ�����ʱ������������Լ�Ϊ�˱����ͷ��м������
	������������ѭ�����ã������Ҫ�ֶ��ͷš�
2. ����������

3. ����������

*******************************************************/




}


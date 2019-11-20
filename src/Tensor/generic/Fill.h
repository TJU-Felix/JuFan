#pragma once
#include "General.h"
#include "Math.h"
/*
��������һ�ε�ַ����ͬһ��������ʹ�����������л�
*/
namespace zf {
	//��䳣��
	template<class T1, class T2>
	void fill_constant(T1* ptr, Int64 size, T2 value)
	{
		for (int i = 0; i < size; i++)
			ptr[i] = value;
	}

	//����˹�����
	template<class T>
	void fill_randnormal(T* ptr, Int64 size, double mean, double std)
	{
		for (int i = 0; i < size; i++)
			ptr[i] = normal(mean, std);
	}

	//�����������
	template<class T>
	void fill_rand(T* ptr, size_t size, double low, double high)
	{

	}

	//����������
	//template<class T>
	//void fill_randint(T* ptr, size_t size, Int64 low, Int64 high);

}
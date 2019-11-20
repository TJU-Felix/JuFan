#pragma once
#include "General.h"
#include "Math.h"
/*
对连续的一段地址进行同一个操作，使用了向量并行化
*/
namespace zf {
	//填充常数
	template<class T1, class T2>
	void fill_constant(T1* ptr, Int64 size, T2 value)
	{
		for (int i = 0; i < size; i++)
			ptr[i] = value;
	}

	//填充高斯随机数
	template<class T>
	void fill_randnormal(T* ptr, Int64 size, double mean, double std)
	{
		for (int i = 0; i < size; i++)
			ptr[i] = normal(mean, std);
	}

	//填充均匀随机数
	template<class T>
	void fill_rand(T* ptr, size_t size, double low, double high)
	{

	}

	//填充随机整数
	//template<class T>
	//void fill_randint(T* ptr, size_t size, Int64 low, Int64 high);

}
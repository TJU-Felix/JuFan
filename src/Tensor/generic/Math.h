#pragma once
/*
提供了一些常用的数学函数
*/
#ifndef PI
#define PI 3.1415926
#endif
#include <random>
namespace {
	double normal(double mean=0, double std=1)
	{
		int i, a;
		double f;
		double uni[2];
		for (i = 0; i < 2; i++)
		{
			a = std::rand() + 689;  //加上689是因为系统产生随机数的更换频率远远不及程序调用函数的时间
			a = a % 1000;
			f = (double)a;
			f = f / 1000.0;
			uni[i] = f;
		}
		double A = std::sqrt((-2)*std::log(uni[0]));
		double B = 2 * PI*uni[1];
		double C = A * std::cos(B);
		double r = mean + C * std;
		return r;
	}
}
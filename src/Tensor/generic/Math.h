#pragma once
/*
�ṩ��һЩ���õ���ѧ����
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
			a = std::rand() + 689;  //����689����Ϊϵͳ����������ĸ���Ƶ��ԶԶ����������ú�����ʱ��
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
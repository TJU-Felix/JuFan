#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <math.h>
#include "TensorStorage.h"
#include "generic/Exception.h"
/*
����
*/
namespace zf {
	class Tensor
	{
		//typedef std::vector<Int64> Shape;
	protected:
		Shape _stride;			//�Ǳ�ƫ�� 2
		Shape _shape;			//��״ 4
		TensorStorage _storage;	//�ײ�洢 5

		int _auto_set_by_shape();//�Զ�������״������������
	public:
		/********************************************
		���캯����*
		�����������ù�������
		�������壬�����Ҫ���������copy����
		*********************************************/
		Tensor() = default;
		Tensor(Tensor&& t) = default;
		Tensor(const Tensor& t) = default;
		Tensor(const TensorStorage& storage, Shape shape);
		//�ͷųɿ�����
		void release();
		/********************************************
		����������**
		��ֵ���ž�Ϊ��������
		��������copy����
		*********************************************/
		Tensor& operator=(const Tensor& t) = default;
		Tensor copy() const;
		int copy_(const Tensor& rhs);
		/********************************************
		ֻ�����������ԣ�*
		*********************************************/
		const Shape& shape() const;
		Size size() const;
		void display_information() const;//TODO
		void display() const;
		TensorStorage storage_unsafe() const;
		bool isNULL() const;//�Ƿ��ǿ�����
		bool isScalar() const;//�Ƿ�������
		bool isVector() const;//�Ƿ���һά����
		bool isTensor() const;//�Ƿ��Ƕ�ά����
		bool isMatrix() const;//�Ƿ��Ƕ�ά����
		/********************************************
		�޸ģ���Ӱ��ײ�洢��ֻӰ��۲�ģʽ��**
		*********************************************/
		Tensor operator=(DataType value);
		Tensor reshape(const Shape& shape) const;
		Tensor& reshape_(const Shape& shape);
		/********************************************
		��������Ƭ����Ӱ��ײ�洢����Ӱ��۲�ģʽ��****
		*********************************************/
		Tensor operator[](Rank rank) const;
		Tensor slice(Rank dim, Int64 begin, Int64 end, Int64 stride = 1) const;//TODO

		/*
		ǿ��ת��
		*/
		operator double() const;
		operator Int64() const;
		//���帺��
		Tensor operator-() const;
		//
		Tensor sum() const;
		Tensor max() const;
		Tensor min() const;

		/********************************************
		���㣺*****
		TensorFunctions.h�ṩ�˸�ȫ�������
		*********************************************/

	};
}


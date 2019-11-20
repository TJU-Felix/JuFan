#pragma once
#include "LayerImpl.hpp"
#include <memory>
/*
����
*/
class LinearLayer :public LayerImpl {
	
public:
	//���캯��
	LinearLayer(int n_input, int n_output)//�������
	{
		parameters.reset({ {1,n_input + 1,n_output },4 });//��������ͬʱ���õ���
	}
	//ǰ�򴫲�
	int Update() override//����(������)����
	{
		parameters.update(lr);
		return 0;
	}
	int ClearGrad() override//��յ���
	{
		parameters.clear_grad();
		return 0;
	}
	int Forward() override//ǰ�򴫲�
	{
		return 0;
	}
	int BackProp() override//���򴫲�
	{
		return 0;
	}
	//����ѧϰ���ʡ���ʼ����ʽ���Ƿ񶳽�
	//...
};
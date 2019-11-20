#pragma once
#include "../Tensor/TensorImpl.hpp"
#include "../Variable/Variable.hpp"
#include "LayerLink.hpp"
#include <memory>

class LayerImpl {
protected:
	std::shared_ptr<LayerImpl> _input_layer;//����Ĳ�
	Variable _output;//���������
	Variable _parameters;//����
	double lr;//ѧϰ����
public:
	LayerImpl() :
		_input_layer(NULL), _output(), _parameters(), lr (0.001){};
	
	/*
	����Ҫ���ĸ��ӿ�
	*/
	virtual int Update()//���²���
	{
		return 0;
	}
	virtual int ClearGrad()//��յ���
	{
		return 0;
	}
	virtual int Forward()//ǰ�򴫲�
	{
		return 0;
	}
	virtual int BackProp()//���򴫲�
	{
		return 0;
	}
	//����
	int set_lr(double LR)//����ѧϰ����
	{
		lr = LR;
		return 0;
	}
	int set_require_grad(bool require)//���ò����Ƿ���Ҫ����
	{
		return _parameters.set_requires_grad(require);
	}
	//�������Ĳ�����
	void operator()(LayerImpl* other)
	{

	}
	template<typename ...Layers>
	void operator()(LayerImpl* other, Layers... others)
	{

	}
};

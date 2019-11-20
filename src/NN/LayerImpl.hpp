#pragma once
#include "../Tensor/TensorImpl.hpp"
#include "../Variable/Variable.hpp"
#include "LayerLink.hpp"
#include <memory>

class LayerImpl {
protected:
	std::shared_ptr<LayerImpl> _input_layer;//输入的层
	Variable _output;//输出的向量
	Variable _parameters;//参数
	double lr;//学习速率
public:
	LayerImpl() :
		_input_layer(NULL), _output(), _parameters(), lr (0.001){};
	
	/*
	最重要的四个接口
	*/
	virtual int Update()//更新参数
	{
		return 0;
	}
	virtual int ClearGrad()//清空导数
	{
		return 0;
	}
	virtual int Forward()//前向传播
	{
		return 0;
	}
	virtual int BackProp()//反向传播
	{
		return 0;
	}
	//设置
	int set_lr(double LR)//设置学习速率
	{
		lr = LR;
		return 0;
	}
	int set_require_grad(bool require)//设置参数是否需要跟新
	{
		return _parameters.set_requires_grad(require);
	}
	//与其他的层连接
	void operator()(LayerImpl* other)
	{

	}
	template<typename ...Layers>
	void operator()(LayerImpl* other, Layers... others)
	{

	}
};

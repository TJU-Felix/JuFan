#pragma once
#include "LayerImpl.hpp"
#include <memory>
/*
线性
*/
class LinearLayer :public LayerImpl {
	
public:
	//构造函数
	LinearLayer(int n_input, int n_output)//输入输出
	{
		parameters.reset({ {1,n_input + 1,n_output },4 });//设置数据同时设置导数
	}
	//前向传播
	int Update() override//更新(参数层)参数
	{
		parameters.update(lr);
		return 0;
	}
	int ClearGrad() override//清空导数
	{
		parameters.clear_grad();
		return 0;
	}
	int Forward() override//前向传播
	{
		return 0;
	}
	int BackProp() override//反向传播
	{
		return 0;
	}
	//设置学习速率、初始化方式、是否冻结
	//...
};
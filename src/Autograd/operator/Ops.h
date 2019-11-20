#pragma once
#include "OperatorHandle.h"
#include <functional>
namespace ai {
namespace op {
	//virtual void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, zf::Tensor& result) = 0;
	//virtual void backward(const zf::Tensor& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) = 0;
	/*
	class Add final:public VariableOperator {
	public:
		void forward(const std::vector<std::shared_ptr<zf::Tensor>>& parent, zf::Tensor& result) const override
		{
			result = *parent[0] + *parent[1];
		}
		void backward(const zf::Tensor& grad, std::vector<std::shared_ptr<zf::Tensor>>& parent) const override
		{
			parent[0].reset(&grad);
			parent[1].reset(&grad);
		}
	}add;
	static class Sum final :public VariableOperator {
	public:
		void forward(const std::vector<std::shared_ptr<zf::Tensor>>& parent, zf::Tensor& result) const override
		{
			result = zf::zeros(parent[0]->shape());
			for (auto &i : parent)
				result += *i;
		}
		void backward(const zf::Tensor& grad, std::vector<std::shared_ptr<zf::Tensor>>& parent) const override
		{
			for (auto &i : parent)
				i.reset(&grad);
		}
	}sum;
	static class Sub final :public VariableOperator {
	public:
		void forward(const std::vector<std::shared_ptr<zf::Tensor>>& parent, zf::Tensor& result) const override
		{
			result = *parent[0] - *parent[1];
		}
		void backward(const zf::Tensor& grad, std::vector<std::shared_ptr<zf::Tensor>>& parent) const override
		{
			parent[0]->add_grad(grad.copy(), true);
			parent[1]->add_grad(0-grad, true);
		}
	}sub;
	static class Mul final :public VariableOperator {
	public:
		void forward(const std::vector<std::shared_ptr<zf::Tensor>>& parent, zf::Tensor& result) const override
		{
			result = *parent[0] * *parent[1];
		}
		void backward(const zf::Tensor& grad, std::vector<std::shared_ptr<zf::Tensor>>& parent) const override
		{
			parent[0].reset(new zf::Tensor(grad**parent[1]));
			parent[1].reset(new zf::Tensor(grad**parent[0]));
		}
	}mul;*/


}//namespace op
}//namespace ai
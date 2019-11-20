#pragma once
#include <vector>
#include <functional>
#include <string>
#include "../../Tensor/Tensor.h"
#include "../generic/common.h"
//#include "../Variable.h"
namespace ai {
	class _Variable_;
	namespace op {
		class VariableOperator
		{
		public:
			virtual std::string name() const =0;
			virtual void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const = 0;
			virtual void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const= 0;
		};
		class Add:public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		}; 
		class Sum :public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Sub :public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Mul :public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Div :public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Sum_self :public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Slice :public VariableOperator
		{
			std::vector<int> _slice_info;
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Index :public VariableOperator
		{
			int _index_info;
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Max_self :public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		class Min_self :public VariableOperator
		{
		public:
			std::string name() const override;
			void forward(const std::vector<std::shared_ptr<ai::_Variable_>>& parent, ai::_Variable_& result) const override;
			void backward(const ai::_Variable_& grad, std::vector<std::shared_ptr<ai::_Variable_>>& parent) const override;
		};
		extern std::shared_ptr<Add> add;
		extern std::shared_ptr<Sum> sum;
		extern std::shared_ptr<Sub> sub;
		extern std::shared_ptr<Mul> mul;
		extern std::shared_ptr<Div> div;
		extern std::shared_ptr<Sum_self> sum_self;
		extern std::shared_ptr<Max_self> max_self;
		extern std::shared_ptr<Min_self> min_self;
		//extern Index index;
		//extern Slice slice;
	}
}
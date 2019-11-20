#include "Variable.h"
namespace ai {
	_Variable_::_Variable_(std::shared_ptr<ai::op::VariableOperator> op, std::vector<std::shared_ptr<_Variable_>> parent, bool training_, bool required_grad_):
		creater(op), training(training_), required_grad(required_grad_)
	{
		creater->forward(parent, *this);
		if (1) {//则设置引用来维持中间变量
			_parent_holder = parent;
			for (auto &i : parent)
				i->_children_holder.push_back(std::shared_ptr<_Variable_>(this));
		}
	}
	_Variable_::_Variable_(const zf::Tensor& tensor, bool training_, bool required_grad_) :
		creater(NULL), training(training_), required_grad(required_grad_)
	{
		_data = tensor;
	};

	void _Variable_::add_grad(zf::Tensor grad_, bool safe_copy)
	{
		_ASSERT(grad_.shape() == _data.shape());
		if (_grad.isNULL()) {
			if (safe_copy)
				_grad = grad_;
			else
				_grad = grad_.copy();
		}
		else {
			_grad += grad_;
		}
		_grad_count++;
	}
	void _Variable_::zero_grad()//清空导数
	{
		if (!_grad.isNULL())
			_grad = 0;
		_grad_count = 0;
	}
	void _Variable_::release_grad()//释放导数
	{
		_grad.release();
	}
	void _Variable_::release_hold()//释放中间变量 unsafe!!
	{
		_children_holder.clear();
		
	}
	const zf::Tensor&  _Variable_::data() const
	{
		return _data;
	}
	zf::Tensor&  _Variable_::data()
	{
		return _data;
	}
	const zf::Tensor&  _Variable_::grad() const
	{
		return _grad;
	}
	int  _Variable_::grad_count() const
	{
		return _grad_count;
	}
	const std::vector<std::shared_ptr<_Variable_>>& _Variable_::parent() const
	{
		return _parent_holder;
	}
	void _Variable_::backward()
	{
		if (_parent_holder.size() == 0)
			return;
		creater->backward(*this, _parent_holder);
	}
}
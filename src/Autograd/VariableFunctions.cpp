#include "VariableFunctions.h"
#include "Variable.h"
#include "VariableFactory.h"
#include "operator/OperatorHandle.h"
namespace ai {
	/* demo
		Tensor operator+(const Tensor& left, const Tensor& right)
		{
			//create the ophandle
			OperatorHandle ophandle(&op_add, "add");
			//set the parter
			ophandle.append_parter(left.variable_unsafe());
			ophandle.append_parter(right.variable_unsafe());
			//create a Variable by the ophandle
			std::shared_ptr<_Variable_> v(new _Variable_(ophandle));
			//make the variable as left and right's child
			left.variable_unsafe()->hold(v);
			right.variable_unsafe()->hold(v);
			return v;
		}
	*/
	using namespace ai::op;
	Tensor operator+(const Tensor& left, const Tensor& right)
	{
		std::vector<std::shared_ptr<_Variable_>> parent({ (left.variable_unsafe()),(right.variable_unsafe()) });
		std::shared_ptr<_Variable_> v(new _Variable_(add, parent, false));
		return Tensor(v);
	}
	Tensor operator+(const DataType& left, const Tensor& right)
	{
		return constant(right.shape(), left) + right;
	}
	Tensor operator+(const Tensor& left, const DataType& right)
	{
		return right + left;
	}
	Tensor operator-(const Tensor& left, const Tensor& right)
	{
		std::vector<std::shared_ptr<_Variable_>> parent({ (left.variable_unsafe()),(right.variable_unsafe()) });
		std::shared_ptr<_Variable_> v(new _Variable_(sub, parent, false));
		return Tensor(v);
	}
	Tensor operator-(const DataType& left, const Tensor& right)
	{
		return constant(right.shape(), left) - right;
	}
	Tensor operator-(const Tensor& left, const DataType& right)
	{
		return left + (-right);
	}
	Tensor operator*(const Tensor& left, const Tensor& right)
	{
		std::vector<std::shared_ptr<_Variable_>> parent({ (left.variable_unsafe()),(right.variable_unsafe()) });
		std::shared_ptr<_Variable_> v(new _Variable_(mul, parent, false));
		return Tensor(v);
	}
	Tensor operator*(const DataType& left, const Tensor& right)
	{
		return constant(right.shape(), left) * right;
	}
	Tensor operator*(const Tensor& left, const DataType& right)
	{
		return right * left;
	}
	Tensor operator/(const Tensor& left, const Tensor& right)
	{
		std::vector<std::shared_ptr<_Variable_>> parent({ (left.variable_unsafe()),(right.variable_unsafe()) });
		std::shared_ptr<_Variable_> v(new _Variable_(div, parent, false));
		return Tensor(v);
	}
	Tensor operator/(const DataType& left, const Tensor& right)
	{
		return constant(right.shape(), left) / right;
	}
	Tensor operator/(const Tensor& left, const DataType& right)
	{
		return left / constant(left.shape(), right);
	}
}
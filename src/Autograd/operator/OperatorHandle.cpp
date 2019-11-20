#if 1
#include "OperatorHandle.h"
#include "../Variable.h"
namespace ai {
	namespace op {
		std::shared_ptr<Add> add(new Add);
		std::shared_ptr<Sum> sum(new Sum);
		std::shared_ptr<Sub> sub(new Sub);
		std::shared_ptr<Mul> mul(new Mul);
		std::shared_ptr<Div> div(new Div);
		std::shared_ptr<Sum_self> sum_self(new Sum_self);
		std::shared_ptr<Max_self> max_self(new Max_self);
		std::shared_ptr<Min_self> min_self(new Min_self);
		//Index index;
		//Slice slice;
		typedef std::vector<std::shared_ptr<ai::_Variable_>> bottom;
		typedef ai::_Variable_ top;
		std::string Add::name() const
		{
			return "[Add]";
		}
		void Add::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data() + parent[1]->data();
		}
		void Add::backward(const top& grad, bottom& parent) const
		{
			parent[0]->add_grad(grad.grad().copy());
			parent[1]->add_grad(grad.grad().copy());
		}
		std::string Sub::name() const
		{
			return "[Sub]";
		}
		void Sub::forward(const bottom& parent, top& result) const
		{	
			/*
			try catch
			*/
			result.data() = parent[0]->data() - parent[1]->data();
		}
		void Sub::backward(const top& grad, bottom& parent) const
		{
			parent[0]->add_grad(grad.grad());
			parent[1]->add_grad(-grad.grad());
		}
		std::string Sum::name() const
		{
			return "[Add]";
		}
		void Sum::forward(const bottom& parent, top& result) const
		{
			/*
			try catch
			*/
			result.data() = parent[0]->data().copy();
			for (int i = 1; i < parent.size(); i++)
				result.data() += parent[i]->data();
		}
		void Sum::backward(const top& grad, bottom& parent) const
		{
			/*
			try catch
			*/
			for (auto &i : parent)
				i->add_grad(grad.grad().copy());
		}
		std::string Mul::name() const
		{
			return "[Mul]";
		}
		void Mul::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data() * parent[1]->data();
		}
		void Mul::backward(const top& grad, bottom& parent) const
		{
			parent[0]->add_grad(grad.grad()*parent[1]->data());
			parent[1]->add_grad(grad.grad()*parent[0]->data());
		}
		std::string Div::name() const
		{
			return "[Div]";
		}
		void Div::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data() / parent[1]->data();
		}
		void Div::backward(const top& grad, bottom& parent) const
		{
			parent[0]->add_grad(grad.grad() / parent[1]->data());
			parent[1]->add_grad(-grad.grad()*parent[0]->data() / (parent[1]->data()*parent[1]->data()));
		}
		std::string Sum_self::name() const
		{
			return "[Sum]";
		}
		void Sum_self::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data().sum();
		}
		void Sum_self::backward(const top& grad, bottom& parent) const
		{
			DataType g = DataType(grad.grad());
			parent[0]->add_grad(zf::constant(parent[0]->data().shape(), g));
		}
		std::string Min_self::name() const
		{
			return "[Min]";
		}
		void Min_self::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data().min();
		}
		void Min_self::backward(const top& grad, bottom& parent) const
		{
			DataType g = DataType(grad.grad());
			parent[0]->add_grad(zf::constant(parent[0]->data().shape(), g));
		}
		std::string Max_self::name() const
		{
			return "[Max]";
		}
		void Max_self::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data().max();
		}
		void Max_self::backward(const top& grad, bottom& parent) const
		{
			DataType g = DataType(grad.grad());
			parent[0]->add_grad(zf::constant(parent[0]->data().shape(), g));
		}
		std::string Slice::name() const
		{
			return "[Slice]";
		}
		void Slice::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data().sum();
		}
		void Slice::backward(const top& grad, bottom& parent) const
		{
			DataType g = DataType(grad.grad());
			parent[0]->add_grad(zf::constant(parent[0]->data().shape(), g));
		}
		std::string Index::name() const
		{
			return "[Index]";
		}
		void Index::forward(const bottom& parent, top& result) const
		{
			result.data() = parent[0]->data().sum();
		}
		void Index::backward(const top& grad, bottom& parent) const
		{
			DataType g = DataType(grad.grad());
			parent[0]->add_grad(zf::constant(parent[0]->data().shape(), g));
		}
	}
}
#endif
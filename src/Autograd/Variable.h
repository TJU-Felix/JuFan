#pragma once
#include "../Tensor/Tensor.h"
#include "operator/OperatorHandle.h"
#include <vector>
/*
存储计算图节点的数据
*/
namespace ai {
	class Tensor;
	class _Variable_ {
		friend class Tensor;
	protected:
		/*******************************************************
		存储数据。包括_data和导数_grad，注意是否求导由外部Tensor控制
		*******************************************************/
		//数据
		zf::Tensor _data;
		zf::Tensor _grad;
		int _grad_count{ 0 };//记录添加了几次导数
		/*******************************************************
		结构数据
		操作符指针：指向得到该结点所进行的操作
		父节点：指向参与运算的结点，反向传播时需要手动释放
		子节点：指向参与了哪些结点的运算，反向传播时需要手动释放
		*******************************************************/
		std::shared_ptr<ai::op::VariableOperator> creater;//得到该张量所需的操作，构造时就已经确定
		std::vector<std::shared_ptr<_Variable_>> _parent_holder;
		std::vector<std::shared_ptr<_Variable_>> _children_holder;//子节点，该张量参与了哪些variable的创建，这里是为了维护中间变量
	public:
		bool required_grad;
		bool training;
		/*******************************************************
		构造函数
		*******************************************************/
		_Variable_(std::shared_ptr<ai::op::VariableOperator> op, std::vector<std::shared_ptr<_Variable_>>, bool training=true, bool required_grad=false);//【用于中间变量的构造】传递操作符，自动计算_data
		_Variable_(const zf::Tensor& tensor, bool training=true, bool required_grad=false);//【用于输入值和参数的构造】传递张量，进行引用作为_data
		~_Variable_()
		{
			_parent_holder.clear();
			_children_holder.clear();
		}
		/*******************************************************
		禁止拷贝
		*******************************************************/
		_Variable_(const _Variable_&) = delete;
		_Variable_(_Variable_&&) = delete;
		_Variable_ operator=(const _Variable_&) = delete;
		/*******************************************************
		动态操作（对外接口）
		********************************************************/
		void add_grad(zf::Tensor grad_, bool safe=true);//添加导数，如果没有导数张量则进行构造
		void zero_grad();//清空导数
		void release_grad();//释放导数
		void release_hold();//释放中间变量
		/*******************************************************
		只读操作（对外接口）
		********************************************************/
		const zf::Tensor& data() const;
		zf::Tensor& data();
		const zf::Tensor& grad() const;
		int grad_count() const;//获取梯度累加次数
		const std::vector<std::shared_ptr<_Variable_>>& parent() const;//获取父节点
		void backward();//将自身的梯度累加到父节点上
	};
}
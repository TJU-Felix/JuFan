#pragma once
#include <vector>
#include <iostream>
#include <queue>
#include <memory>
#include "../Tensor/Tensor.h"
#include "Variable.h"


namespace ai {
	/*
	将Variable封装，便于构建计算图
	*/
	class Tensor {
	protected:
		/*******************************************************
		存储数据。包括底层数据和是否要求导数的标志位
		*******************************************************/
		std::shared_ptr<_Variable_> _impl;//实际的存储数据
	public:
		/*******************************************************
		构造函数，默认的拷贝构造。
		如果要创建各种类型的张量，请使用TensorFactory提供的函数
		*******************************************************/
		Tensor() :_impl(NULL) {};
		Tensor(Tensor&& rhs) = default;
		Tensor(const Tensor& rhs) = default;
		Tensor& operator=(const Tensor& rhs) = default;
		Tensor(std::shared_ptr<_Variable_> impl, bool required_ = false) :_impl(impl) { _impl->required_grad = required_; };
		
		Shape shape() const
		{
			return _impl->data().shape();
		}
		/*******************************************************
		debug
		*******************************************************/
		void display()
		{
			using std::cout;
			using std::endl;
			if (_impl == NULL) {
				cout << "NULL Tensor" << endl;
				return;
			}
			_impl->data().display();
			cout << "required_grad: " << (_impl->required_grad ? "true" : "false") << endl;
			if (_impl->creater)
				cout << _impl->creater->name() << endl;
			cout << " grad: ";
			_impl->grad().display();
			cout << endl;
		}
		/*******************************************************
		不安全的访问操作，访问底层数据。只可用于运算符函数
		*******************************************************/
		inline std::shared_ptr<_Variable_> variable_unsafe() const
		{
			return _impl;
		}

		inline zf::Tensor grad() const
		{
			return _impl->grad();
		}

		/*******************************************************
		反向传播。积累梯度值，销毁中间变量
		*******************************************************/
		void backward() const
		{
			if (!isScalar())
				/*报错*/return;
			else {
				//_impl->creater.backward(_impl->_data);
			}
			_impl->_grad = zf::scalar(1);//设置初始导数
			//广度优先搜索
			std::queue<std::shared_ptr<_Variable_>> buffer;
			buffer.push(_impl);
			while (!buffer.empty())
			{
				//弹出队首元素
				auto front = buffer.front();
				buffer.pop();
				//执行反向传播。将自身的梯度值传播到父节点上
				front->backward();
				//调用钩子函数
				//front->hook();
				
				//将父节点入队列
				auto parent = front->parent();
				for (auto &i : parent) {
					buffer.push(i);
				}//销毁维持的中间变量
				//front->release_hold();
			}
		}
		inline bool isScalar() const//只有是标量才能进行反向传播
		{
			return _impl->_data.isScalar();
		}

		Tensor sum()
		{
			std::vector<std::shared_ptr<_Variable_>> parent({ _impl });
			std::shared_ptr<_Variable_> v(new _Variable_(ai::op::sum_self, parent, false));
			return Tensor(v);
		}
	};
}
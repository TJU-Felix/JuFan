#pragma once
#include <vector>
#include <iostream>
#include <queue>
#include <memory>
#include "../Tensor/Tensor.h"
#include "Variable.h"


namespace ai {
	/*
	��Variable��װ�����ڹ�������ͼ
	*/
	class Tensor {
	protected:
		/*******************************************************
		�洢���ݡ������ײ����ݺ��Ƿ�Ҫ�����ı�־λ
		*******************************************************/
		std::shared_ptr<_Variable_> _impl;//ʵ�ʵĴ洢����
	public:
		/*******************************************************
		���캯����Ĭ�ϵĿ������졣
		���Ҫ�����������͵���������ʹ��TensorFactory�ṩ�ĺ���
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
		����ȫ�ķ��ʲ��������ʵײ����ݡ�ֻ���������������
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
		���򴫲��������ݶ�ֵ�������м����
		*******************************************************/
		void backward() const
		{
			if (!isScalar())
				/*����*/return;
			else {
				//_impl->creater.backward(_impl->_data);
			}
			_impl->_grad = zf::scalar(1);//���ó�ʼ����
			//�����������
			std::queue<std::shared_ptr<_Variable_>> buffer;
			buffer.push(_impl);
			while (!buffer.empty())
			{
				//��������Ԫ��
				auto front = buffer.front();
				buffer.pop();
				//ִ�з��򴫲�����������ݶ�ֵ���������ڵ���
				front->backward();
				//���ù��Ӻ���
				//front->hook();
				
				//�����ڵ������
				auto parent = front->parent();
				for (auto &i : parent) {
					buffer.push(i);
				}//����ά�ֵ��м����
				//front->release_hold();
			}
		}
		inline bool isScalar() const//ֻ���Ǳ������ܽ��з��򴫲�
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
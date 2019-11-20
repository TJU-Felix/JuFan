#pragma once
#include "../Tensor/Tensor.h"
#include "operator/OperatorHandle.h"
#include <vector>
/*
�洢����ͼ�ڵ������
*/
namespace ai {
	class Tensor;
	class _Variable_ {
		friend class Tensor;
	protected:
		/*******************************************************
		�洢���ݡ�����_data�͵���_grad��ע���Ƿ������ⲿTensor����
		*******************************************************/
		//����
		zf::Tensor _data;
		zf::Tensor _grad;
		int _grad_count{ 0 };//��¼����˼��ε���
		/*******************************************************
		�ṹ����
		������ָ�룺ָ��õ��ý�������еĲ���
		���ڵ㣺ָ���������Ľ�㣬���򴫲�ʱ��Ҫ�ֶ��ͷ�
		�ӽڵ㣺ָ���������Щ�������㣬���򴫲�ʱ��Ҫ�ֶ��ͷ�
		*******************************************************/
		std::shared_ptr<ai::op::VariableOperator> creater;//�õ�����������Ĳ���������ʱ���Ѿ�ȷ��
		std::vector<std::shared_ptr<_Variable_>> _parent_holder;
		std::vector<std::shared_ptr<_Variable_>> _children_holder;//�ӽڵ㣬��������������Щvariable�Ĵ�����������Ϊ��ά���м����
	public:
		bool required_grad;
		bool training;
		/*******************************************************
		���캯��
		*******************************************************/
		_Variable_(std::shared_ptr<ai::op::VariableOperator> op, std::vector<std::shared_ptr<_Variable_>>, bool training=true, bool required_grad=false);//�������м�����Ĺ��졿���ݲ��������Զ�����_data
		_Variable_(const zf::Tensor& tensor, bool training=true, bool required_grad=false);//����������ֵ�Ͳ����Ĺ��졿��������������������Ϊ_data
		~_Variable_()
		{
			_parent_holder.clear();
			_children_holder.clear();
		}
		/*******************************************************
		��ֹ����
		*******************************************************/
		_Variable_(const _Variable_&) = delete;
		_Variable_(_Variable_&&) = delete;
		_Variable_ operator=(const _Variable_&) = delete;
		/*******************************************************
		��̬����������ӿڣ�
		********************************************************/
		void add_grad(zf::Tensor grad_, bool safe=true);//��ӵ��������û�е�����������й���
		void zero_grad();//��յ���
		void release_grad();//�ͷŵ���
		void release_hold();//�ͷ��м����
		/*******************************************************
		ֻ������������ӿڣ�
		********************************************************/
		const zf::Tensor& data() const;
		zf::Tensor& data();
		const zf::Tensor& grad() const;
		int grad_count() const;//��ȡ�ݶ��ۼӴ���
		const std::vector<std::shared_ptr<_Variable_>>& parent() const;//��ȡ���ڵ�
		void backward();//��������ݶ��ۼӵ����ڵ���
	};
}
#pragma once
#include "TensorView.h"
#include "generic/General.h"
#include "generic/Fill.h"
namespace zf {

	template<class T>
	struct _Storage_ {
		T* data;
		Int64 size;
		_Storage_(Int64 size_) :size(size_)
		{
			data = new T[size];
		}
		~_Storage_()
		{
			delete[]data;
		}
	};

	class TensorStorage {
	private:
		//�Եײ����ݽ���һ���װ
		ViewStorage _view;
		//��ײ�����ݴ洢ָ��
		std::shared_ptr<_Storage_<DataType>> _data;
		//���ݹ�ģ
		Int64 _size;
	public:
		//���캯���������µĵײ�ָ��
		TensorStorage(Int64 size)
		{
			if (size)
				_data.reset(new _Storage_<DataType>(_size = size));
			else {
				_data = NULL;
				_size = 0;
			}
		}
		TensorStorage() :_size(0), _data(NULL) {};
		//�������壬����ײ�
		TensorStorage(const TensorStorage& rhs) = default;
		TensorStorage(TensorStorage&& rhs) = default;
		TensorStorage& operator=(const TensorStorage& rhs) = default;
		//�����������ݣ����ù۲�ģʽ
		TensorStorage copy() const
		{
			TensorStorage temp(_size);
			temp.copy_(*this);
			return temp;
		}
		//�ͷŵײ�
		void release()
		{
			_size = 0;
			_data = NULL;
			_view = ViewStorage();
		}

		//����View������������
		inline void copy_(const TensorStorage&rhs)
		{
			for (int i = 0; i < _size; i++)
				(*this)[i] = rhs[i];
		}
		//��Ƭ������ײ����ݣ��ı�view
		TensorStorage slice(Rank base, Rank stride, Rank per_stride, Rank persize, Rank n_each_segment, Rank n_segment) const
		{
			TensorStorage temp(*this);
			temp._view.append(_View_Rank(base, stride, per_stride, n_segment, n_each_segment, persize));
			temp._size = persize * n_each_segment*n_segment;
			return temp;
		}
		
		//����
		inline DataType& operator[](Int64 rank) const
		{
			return (*_data).data[_view(rank)];
		}

		//��ֵ
		inline TensorStorage& operator=(DataType value)
		{
			for (int i = 0; i < _size; i++)
				(*this)[i] = value;
			return *this;
		}
		//����
		template<class VST>
		inline void traver(const VST& visit)
		{
			for (int i = 0; i < _size; i++)
				(*this)[i] = visit((*this)[i]);
		}


		//���ʵײ�ָ�룬 ���ڸ�ֵ
		inline DataType* bottom_dataptr_unsafe() const
		{
			return _data->data;
		}
		inline Int64 bottom_size() const
		{
			return _data->size;
		}
		inline Int64 size() const
		{
			return _size;
		}
	};


}

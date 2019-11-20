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
		//对底层数据进行一层封装
		ViewStorage _view;
		//最底层的数据存储指针
		std::shared_ptr<_Storage_<DataType>> _data;
		//数据规模
		Int64 _size;
	public:
		//构造函数，创建新的底层指针
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
		//引用语义，共享底层
		TensorStorage(const TensorStorage& rhs) = default;
		TensorStorage(TensorStorage&& rhs) = default;
		TensorStorage& operator=(const TensorStorage& rhs) = default;
		//拷贝顶层数据，重置观察模式
		TensorStorage copy() const
		{
			TensorStorage temp(_size);
			temp.copy_(*this);
			return temp;
		}
		//释放底层
		void release()
		{
			_size = 0;
			_data = NULL;
			_view = ViewStorage();
		}

		//根据View拷贝顶层数据
		inline void copy_(const TensorStorage&rhs)
		{
			for (int i = 0; i < _size; i++)
				(*this)[i] = rhs[i];
		}
		//切片，共享底层数据，改变view
		TensorStorage slice(Rank base, Rank stride, Rank per_stride, Rank persize, Rank n_each_segment, Rank n_segment) const
		{
			TensorStorage temp(*this);
			temp._view.append(_View_Rank(base, stride, per_stride, n_segment, n_each_segment, persize));
			temp._size = persize * n_each_segment*n_segment;
			return temp;
		}
		
		//索引
		inline DataType& operator[](Int64 rank) const
		{
			return (*_data).data[_view(rank)];
		}

		//赋值
		inline TensorStorage& operator=(DataType value)
		{
			for (int i = 0; i < _size; i++)
				(*this)[i] = value;
			return *this;
		}
		//遍历
		template<class VST>
		inline void traver(const VST& visit)
		{
			for (int i = 0; i < _size; i++)
				(*this)[i] = visit((*this)[i]);
		}


		//访问底层指针， 用于赋值
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

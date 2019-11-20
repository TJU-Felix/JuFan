#pragma once
#define Address DataType*
#include <vector>
#include "generic/General.h"
namespace zf {
	/*
	提供了针对于张量的观察底层数组的方式
	*/
	struct _View_Rank {
		size_t base, stride_between_segment, stride_in_segment, n_segment ,n_in_segment, persize;
		size_t size;
		_View_Rank(Rank base_,
			Rank stride_between_segment_,
			Rank stride_in_segment_,
			Rank n_segment_,
			Rank n_per_segment_,
			Rank persize_) :
			base(base_),
			stride_between_segment(stride_between_segment_),
			stride_in_segment(stride_in_segment_),
			n_segment(n_segment_), 
			n_in_segment(n_per_segment_), 
			persize(persize_)
		{
			size = persize * n_segment*n_in_segment;
		};
		
		inline Rank operator()(Rank rank) const
		{
			_ASSERT(rank < size);
			size_t m = rank / (persize*n_in_segment), n = rank % (persize*n_in_segment);
			return base + m * stride_between_segment + (n / persize)*stride_in_segment + (n%persize);
		}
	};

	/*
	将多种观察方式组合起来
	*/
	class ViewStorage {
	protected:
		std::vector<_View_Rank> _views;
		int _deep;
	public:
		ViewStorage() :_deep(0) {};
		void append(const _View_Rank& v)
		{
			_views.push_back(v);
			_deep ++;
		}
		inline size_t operator()(Int64 rank) const
		{
			if (_deep == 0)
				return rank;
			size_t R = rank;
			for (int i = _deep - 1; i >= 0; i--) {
				R = _views[i](R);
			}
			return R;
		}

	};
}
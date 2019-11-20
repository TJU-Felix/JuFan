#pragma once
//若数组维度非法，则抛出异常
#define TENSOR_CHECK_SHAPE_INIT(shape) \
	for(int i=0;i<shape.size();i++)if(shape[i]<=0)\
		throw "Illegal Arguments!"

//若两个张量的形状不相等，则抛出异常
#define TENSOR_CHECK_SAME_SHAPE(shape1, shape2)\
	if(shape1.size()==shape2.size()){\
		for(int i=0;i<shape1.size();i++)if(shape1[i]!=shape2[i])throw "Different Shape!";\
	}else{throw "Different Shape!";}

//若该张量不是期望的形状，则抛出异常
#define TENSOR_CHECK_EXPSHAPE(tensor, shape)\
	try{TENSOR_CHECK_SAME_SHAPE(tensor._shape, shape);}catch(...){throw "Wrong Shape!";}

//若两个形状的尺寸大小不相等，则抛出异常
#define TENSOR_CHECK_SAME_SIZE(shape1, shape2)\
	if(shape_to_size(shape1)!=shape_to_size(shape2))throw "Two shape don't have the same size!"

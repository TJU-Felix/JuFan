#pragma once
//������ά�ȷǷ������׳��쳣
#define TENSOR_CHECK_SHAPE_INIT(shape) \
	for(int i=0;i<shape.size();i++)if(shape[i]<=0)\
		throw "Illegal Arguments!"

//��������������״����ȣ����׳��쳣
#define TENSOR_CHECK_SAME_SHAPE(shape1, shape2)\
	if(shape1.size()==shape2.size()){\
		for(int i=0;i<shape1.size();i++)if(shape1[i]!=shape2[i])throw "Different Shape!";\
	}else{throw "Different Shape!";}

//��������������������״�����׳��쳣
#define TENSOR_CHECK_EXPSHAPE(tensor, shape)\
	try{TENSOR_CHECK_SAME_SHAPE(tensor._shape, shape);}catch(...){throw "Wrong Shape!";}

//��������״�ĳߴ��С����ȣ����׳��쳣
#define TENSOR_CHECK_SAME_SIZE(shape1, shape2)\
	if(shape_to_size(shape1)!=shape_to_size(shape2))throw "Two shape don't have the same size!"

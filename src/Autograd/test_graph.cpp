#if 1
#include "Autograd.h"
#include <iostream>
#include <string>
using namespace std;

int main()
{
	/*
	手动搭建神经网络
	*/
	using ai::Tensor;
	Tensor a = ai::rand({ 2,3 }), b = ai::rand({ 2,3 });
	
	auto c = (a + b).sum();
	a.display();
	b.display();
	c.display();
	c.backward();
	a.display();
	b.display();
	c.display();
	//(a + b).display();
	//auto c = a + b;
	//c.display();
	//c.backward();
	getchar();
	return 0;
}
#endif
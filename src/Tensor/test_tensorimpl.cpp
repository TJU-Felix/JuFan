#if 1
#include "Tensor.h"
#include <iostream>
//#include <gdiplus.h>
using namespace std;
#define HINT(message) cout<<#message<<"测试完毕，按任意键继续"<<endl
int main()
{
	if (0) {
		zf::Tensor t = zf::rand({ 2,3,4 });
		t.display();
		t[0].display();
		t[0].display();
		t[0][2].display();
		t[0][2][1].display();
		t[0][2] = 2.33;
		t[0].display();
	}
	if (1) {
		zf::Tensor t1 = zf::rand({ 2,3 });
		zf::Tensor t2 = zf::rand({ 2,3 });
		t1.display();
		t2.display();
		(t1 + t2).display();
	}
	return 0;
}
#endif
#include <windows.h>
#include <math.h>


void main(void)
{
	double x=3.32413567,x2;
	
	//x=(x*1.010000f);
	x=modf( x, &x2 );
	x/=0.0001f;
	printf("x=%f x2=%f",x,x2);
	exit(0);
}
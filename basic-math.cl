/*  vim:set syntax=c: */
#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef double3  Vec3;
typedef double4  Quat;
typedef double16 Mat33; // wasting some bytes here


// 3x3 matrices are row-major (can be changed as long as accessors are used consistently)

// get element
double m33at(Mat33 m, ushort row, ushort col){ return (((double*)(&m))[3*row+col]); }
// set the whole matrix from row-ordered entries
Mat33 m33set(double c00, double c01, double c02, double c10, double c11, double c12, double c20, double c21, double c22){ return (Mat33)(c00,c01,c02, c10,c11,c12, c20,c21,c22, /* tail*/ 0,0,0,0,0,0,0); }


// the following code uses accessors, hence no change if layout changes
// set matrix from columns as vectors
Mat33 m33setCols(Vec3 c0, Vec3 c1, Vec3 c2){ return m33set(c0.s0,c1.s0,c2.s0, c0.s1,c1.s1,c2.s1, c0.s2,c1.s2,c2.s2); }
// set matrix from rows as vectors
Mat33 m33setRows(Vec3 r0, Vec3 r1, Vec3 r2){ return m33set(r0.s0,r0.s1,r0.s2, r1.s0,r1.s1,r1.s2, r2.s0,r2.s1,r2.s2); }
// get row
Vec3  m33row(Mat33 m, ushort row){ return (Vec3)(m33at(m,row,0),m33at(m,row,1),m33at(m,row,2)); }
// get columns
Vec3  m33col(Mat33 m, ushort col){ return (Vec3)(m33at(m,0,col),m33at(m,1,col),m33at(m,2,col)); }
// get diagonal
Vec3  m33diag(Mat33 m){ return (Vec3)(m33at(m,0,0),m33at(m,1,1),m33at(m,2,2)); }
// transpose
Mat33 m33trans(Mat33 m){ return m33setRows(m33col(m,0),m33col(m,1),m33col(m,2)); }
// http://en.wikipedia.org/wiki/Outer_product
Mat33 outerVV3(Vec3 a, Vec3 b){ return m33set(a.s0*b.s0,a.s0*b.s1,a.s0*b.s2,a.s1*b.s0,a.s1*b.s1,a.s1*b.s2,a.s2*b.s0,a.s2*b.s1,a.s2*b.s2); }
// matrix-matrix multiply
Mat33 mulMM3(Mat33 a, Mat33 b){
	Vec3 ar0=m33row(a,0),ar1=m33row(a,1),ar2=m33row(a,2);
	Vec3 bc0=m33col(b,0),bc1=m33col(b,1),bc2=m33col(b,2);
	return m33set(
		dot(ar0,bc0),dot(ar0,bc1),dot(ar0,bc2),
		dot(ar1,bc0),dot(ar1,bc1),dot(ar1,bc2),
		dot(ar2,bc0),dot(ar2,bc1),dot(ar2,bc2)
	);
}
// matrix-vector multiply
Vec3 mulMV3(Mat33 a, Vec3 b){	return (Vec3)(dot(m33row(a,0),b),dot(m33row(a,1),b),dot(m33row(a,2),b)); }



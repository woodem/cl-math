/*  vim:set syntax=c: */
#ifndef _BASIC_MATH_CL_
#define _BASIC_MATH_CL_

#ifndef __OPENCL_VERSION__
	// only types when included from host code
	#include<math.h>
	#include<CL/cl.h>
	#include<CL/cl_platform.h>
	#ifdef __cplusplus
		#include<initializer_list>
		#include<stdexcept>
		#include<cassert>
		#include<iostream>
		// adapter from
		// #include<boost/serialization/strong_typedef.hpp>
		// operators == and < dropped
		// initializer list ctor added to be source-compat with gcc-dialect syntax for vector types in c99
		#define VECTOR_STRONG_TYPEDEF(T,D,numT,/*number of elements for initialization*/N) \
			struct D { \
				typedef numT Scalar; \
				T t; \
				explicit D(const T t_) : t(t_){}; \
				D(){}; \
				D(const D & t_) : t(t_.t){} \
				D(std::initializer_list<Scalar> l){ assert(l.size()==N); /*if(l.size()!=N) throw std::runtime_error("Error assigning " #N " elements to " #D " in initialization");*/ int i=0; for(auto n: l) ((Scalar*)this)[i++]=n; } \
				D & operator=(const D & rhs) { t = rhs.t; return *this;}	\
				D & operator=(const T & rhs) { t = rhs; return *this;}  \
				Scalar& operator[](int ix){ assert(ix>=0 && ix<N); return ((Scalar*)this)[ix]; } \
				const Scalar& operator[](int ix) const { assert(ix>=0 && ix<N); return ((Scalar*)this)[ix]; } \
				operator T & () { return t; } \
			};
		typedef cl_double Real;
		VECTOR_STRONG_TYPEDEF(cl_double3,Vec3,cl_double,3);
		VECTOR_STRONG_TYPEDEF(cl_double4,Quat,cl_double,4);
		VECTOR_STRONG_TYPEDEF(cl_double16,Mat3,cl_double,9);
		static std::ostream& operator<<(std::ostream &os, const Vec3& v){ os<<"("<<v[0]<<","<<v[1]<<","<<v[2]<<")"; return os; }
		static std::ostream& operator<<(std::ostream &os, const Quat& v){ os<<"("<<v[0]<<","<<v[1]<<","<<v[2]<<v[3]<<")"; return os; }
		static std::ostream& operator<<(std::ostream &os, const Mat3& m){ os<<"("<<m[0]<<","<<m[1]<<","<<m[2]<<", "<<m[3]<<","<<m[4]<<","<<m[5]<<", "<<m[6]<<","<<m[7]<<","<<m[8]<<")"; return os; }
	#else
		typedef cl_double Real;
		typedef cl_double3 Vec3;
		typedef cl_double4 Quat;
		typedef cl_double16 Mat3;
	#endif
	// for initializing stuff in ctors from host code
	inline Vec3 Vec3_set(Real x, Real y, Real z){ Vec3 ret={x,y,z}; return ret; }
	inline Mat3 Mat3_set(Real a, Real b, Real c, Real d, Real e, Real f, Real g, Real h, Real i){ Mat3 ret={a,b,c,d,e,f,g,h,i}; return ret; }
	inline Mat3 Mat3_identity(){ return Mat3_set(1,0,0,0,1,0,0,0,1); }
	inline Quat Quat_identity(){ Quat ret={0,0,0,1}; return ret; }
#else

/* code for OpenCL device */

#ifdef cl_khr_fp64
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
	#error "Neiter of cl_khr_fp64/cl_amd_fp64 OpenCL extensions is supported."
#endif

// do we need this?
typedef double Real;
typedef double3  Vec3;
// x,y,z,w order
typedef double4  Quat;
// row-major storage; wasting some bytes here
typedef double16 Mat3; 


// 3x3 matrices are row-major (can be changed as long as accessors are used consistently)
// getters
double Mat3_at(Mat3 m, ushort row, ushort col){ return (((double*)(&m))[3*row+col]); }
Vec3 Mat3_row(Mat3 m, ushort row){ return (Vec3)(Mat3_at(m,row,0),Mat3_at(m,row,1),Mat3_at(m,row,2)); }
Vec3 Mat3_col(Mat3 m, ushort col){ return (Vec3)(Mat3_at(m,0,col),Mat3_at(m,1,col),Mat3_at(m,2,col)); }
Vec3 Mat3_diag(Mat3 m){ return (Vec3)(Mat3_at(m,0,0),Mat3_at(m,1,1),Mat3_at(m,2,2)); }
// setters
Mat3 Mat3_set(double c00, double c01, double c02, double c10, double c11, double c12, double c20, double c21, double c22){ return (Mat3)(c00,c01,c02, c10,c11,c12, c20,c21,c22, /* tail*/ 0,0,0,0,0,0,0); }
// (the following code uses accessors, hence no change if layout changes)
Mat3 Mat3_setCols(Vec3 c0, Vec3 c1, Vec3 c2){ return Mat3_set(c0.s0,c1.s0,c2.s0, c0.s1,c1.s1,c2.s1, c0.s2,c1.s2,c2.s2); }
Mat3 Mat3_setRows(Vec3 r0, Vec3 r1, Vec3 r2){ return Mat3_set(r0.s0,r0.s1,r0.s2, r1.s0,r1.s1,r1.s2, r2.s0,r2.s1,r2.s2); }
Mat3 Mat3_setDiag(Vec3 d){ return Mat3_set(d.s0,0,0, 0,d.s1,0, 0,0,d.s2); }
// const setters
Mat3 Mat3_zero(){ return (Mat3)(0.); }
Mat3 Mat3_identity(){ return Mat3_setDiag((Vec3)1.); }

// operations
Mat3 Mat3_transpose(Mat3 m){ return Mat3_setRows(Mat3_col(m,0),Mat3_col(m,1),Mat3_col(m,2)); }
double Mat3_trace(Mat3 m){ return Mat3_at(m,0,0)+Mat3_at(m,1,1)+Mat3_at(m,2,2); }
double Mat3_det(Mat3 m);
// matrix-vector multiply
Vec3 Mat3_multV(Mat3 a, Vec3 b){	return (Vec3)(dot(Mat3_row(a,0),b),dot(Mat3_row(a,1),b),dot(Mat3_row(a,2),b)); }
// matrix-matrix multiply
Mat3 Mat3_multM(Mat3 a, Mat3 b); // impl below
// Gram-Schmidt orthonormalization with stable 0th column
Mat3 Mat3_orthonorm_col0(Mat3);  // impl below
Mat3 Mat3_orthonorm_row0(Mat3);  // impl below
// set rotation matrix from local x-axis and orient the other two arbitrarily (aligned with global axes)
Mat3 Mat3_rot_setYZ(Vec3 locX);

// http://en.wikipedia.org/wiki/Outer_product
Mat3 Vec3_outer(Vec3 a, Vec3 b){ return Mat3_set(a.s0*b.s0,a.s0*b.s1,a.s0*b.s2,a.s1*b.s0,a.s1*b.s1,a.s1*b.s2,a.s2*b.s0,a.s2*b.s1,a.s2*b.s2); }
// for host code compat
Vec3 Vec3_set(Real x, Real y, Real z){ return (Vec3)(x,y,z); }

Quat Quat_identity(){ return (Quat)(0,0,0,1); }
Quat Quat_conjugate(Quat q){ return (Quat)(-q.x,-q.y,-q.z,q.w); }
Quat Mat3_toQuat(Mat3 rot); // impl below
Vec3 Quat_rotate(Quat q, Vec3 v);
Quat Quat_fromAngleAxis(double angle, Vec3 axis);
Quat Quat_fromRotVec(Vec3 rot){ double n=length(rot); return n>0.?Quat_fromAngleAxis(n,rot/n):Quat_identity(); }
void Quat_toAngleAxis(Quat q, double* angle, Vec3* axis);
Vec3 Quat_toRotVec(Quat q){ double angle; Vec3 axis; Quat_toAngleAxis(q,&angle,&axis); return angle*axis; }

Vec3 Vec3_unitX(){ return (Vec3)(1,0,0); }
Vec3 Vec3_unitY(){ return (Vec3)(0,1,0); }
Vec3 Vec3_unitZ(){ return (Vec3)(0,0,1); }
Vec3 Vec3_unit(ushort i){ return (i==0?Vec3_unitX():(i==1?Vec3_unitY():Vec3_unitZ())); }
Vec3 Vec3_zero() { return (Vec3)(0,0,0); }

double Vec3_sqNorm(Vec3 v){ return dot(v,v); }



/* non-trivial implementations */

Mat3 Mat3_inv(Mat3 m){
	#define _M(i,j) Mat3_at(m,i,j)
	return (1/Mat3_det(m))*Mat3_set(
		+_M(2,2)*_M(1,1)-_M(2,1)*_M(1,2), -_M(2,2)*_M(0,1)+_M(2,1)*_M(0,2), +_M(1,2)*_M(0,1)-_M(1,1)*_M(0,2),
		-_M(2,2)*_M(1,0)+_M(2,0)*_M(1,2), +_M(2,2)*_M(0,0)-_M(2,0)*_M(0,2), -_M(1,2)*_M(0,0)+_M(1,0)*_M(0,2),
		+_M(2,1)*_M(1,0)-_M(2,0)*_M(1,1), -_M(2,1)*_M(0,0)+_M(2,0)*_M(0,1), +_M(1,1)*_M(0,0)-_M(1,0)*_M(0,1)
	);
	#undef _M
}
double Mat3_det(Mat3 m){
	return
		+Mat3_at(m,0,0)*Mat3_at(m,1,1)*Mat3_at(m,2,2)
		+Mat3_at(m,0,1)*Mat3_at(m,1,2)*Mat3_at(m,2,0)
		+Mat3_at(m,0,2)*Mat3_at(m,1,0)*Mat3_at(m,2,1)
		-Mat3_at(m,0,0)*Mat3_at(m,1,2)*Mat3_at(m,2,1)
		-Mat3_at(m,0,1)*Mat3_at(m,1,0)*Mat3_at(m,2,2)
		-Mat3_at(m,0,2)*Mat3_at(m,1,1)*Mat3_at(m,2,0);
}

Mat3 Mat3_multM(Mat3 a, Mat3 b){
	Vec3 ar0=Mat3_row(a,0),ar1=Mat3_row(a,1),ar2=Mat3_row(a,2);
	Vec3 bc0=Mat3_col(b,0),bc1=Mat3_col(b,1),bc2=Mat3_col(b,2);
	return Mat3_set(
		dot(ar0,bc0),dot(ar0,bc1),dot(ar0,bc2),
		dot(ar1,bc0),dot(ar1,bc1),dot(ar1,bc2),
		dot(ar2,bc0),dot(ar2,bc1),dot(ar2,bc2)
	);
}

// following routines implemented following:
// eigen3/Eigen/src/Geometry/Quaternion.h: quaternionbase_assign_impl
Mat3 Quat_toMat3(Quat q){
	double twx=2*q.w*q.x, twy=2*q.w*q.y, twz=2*q.w*q.z,
		txx=2*q.x*q.x, txy=2*q.x*q.y, txz=2*q.x*q.z,
		tyy=2*q.y*q.y, tyz=2*q.y*q.z, tzz=2*q.z*q.z;
	return Mat3_set(
		1-(tyy+tzz),txy-twz,txz+twy,
		txy+twz,1-(txx+tzz),tyz-twx,
		txz-twy,tyz+twx,1-(txx+tyy)
	);
}
/* the matrix must be orthonormal! */
Quat Mat3_toQuat(Mat3 rot){
	double t=Mat3_trace(rot);
	Quat q;
	if(t>0){
		t=sqrt(t+1.);
		q.w=.5*t;
		t=.5/t;
		q.x=(Mat3_at(rot,2,1)-Mat3_at(rot,1,2))*t;
		q.y=(Mat3_at(rot,0,2)-Mat3_at(rot,2,0))*t;
		q.z=(Mat3_at(rot,1,0)-Mat3_at(rot,0,1))*t;
	} else {
		ushort i=0;
		if(Mat3_at(rot,1,1)>Mat3_at(rot,0,0)) i=0;
		if(Mat3_at(rot,2,2)>Mat3_at(rot,i,i)) i=2;
		ushort j=(i+1)%3, k=(i+2)%3;
		t=sqrt(Mat3_at(rot,i,i)-Mat3_at(rot,j,j)-Mat3_at(rot,k,k)+1.);
		((double*)&q)[i]=.5*t;
		t=.5/t;
		q.w=(Mat3_at(rot,k,j)-Mat3_at(rot,j,k))*t;
		((double*)&q)[j]=(Mat3_at(rot,j,i)+Mat3_at(rot,i,j))*t;
		((double*)&q)[k]=(Mat3_at(rot,k,i)+Mat3_at(rot,i,k))*t;
	}
	return q;
}

Mat3 Mat3_orthonorm_col0(Mat3 m){
	Vec3 x=Mat3_col(m,0), y=Mat3_col(m,1);
	x=normalize(x);
	y=y-x*dot(x,y);
	y=normalize(y);
	return Mat3_setCols(x,y,cross(x,y));
}

Mat3 Mat3_orthonorm_row0(Mat3 m){
	Vec3 x=Mat3_row(m,0), y=Mat3_row(m,1);
	x=normalize(x);
	y=y-x*dot(x,y);
	y=normalize(y);
	return Mat3_setRows(x,y,cross(x,y));
}

Mat3 Mat3_rot_setYZ(Vec3 locX){
	Vec3 locY=cross(locX,(fabs(locX.y)<fabs(locX.z))?Vec3_unitY():Vec3_unitZ());
	locY=normalize(locY-locX*dot(locX,locY));
	return Mat3_setRows(locX,locY,cross(locX,locY));
}


Vec3 Quat_rotate(Quat q, Vec3 v){
	Vec3 uv=2*cross(q.xyz,v);
	return v+q.w*uv+cross(q.xyz,uv);
}

Quat Quat_multQ(Quat a, Quat b){
	return (Quat)(
		a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
		a.w*b.y+a.y*b.w+a.z*b.x-a.x*b.z,
		a.w*b.z+a.z*b.w+a.x*b.y-a.y*b.x,
		a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z
	);
}	

Quat Quat_fromAngleAxis(double angle, Vec3 axis){
	Quat ret;
	ret.w=cos(.5*angle);
	ret.xyz=sin(.5*angle)*axis;
	return ret;
}

// #include"print.cl"

void Quat_toAngleAxis(Quat q, double* angle, Vec3* axis){
	/*
	printNumArgs(a,b,c,d);
	printNumArgs(a,b,c);
	printNumArgs(a,b);
	printNumArgs(a);
	*/

	double n2=dot(q.xyz,q.xyz); // squared norm
	if(n2<DBL_EPSILON*DBL_EPSILON){ /* should never happen, since quaternions should be normalized */
		*angle=0.;
		*axis=(Vec3)(1,0,0);
	} else {
		*angle=2*acos(clamp(q.w,-1.,1.));
		*axis=q.xyz/sqrt(n2);
	}
}

#endif /* __OPENCL_VERSION__ */
#endif /* _BASIC_MATH_CL */

import pyopencl as cl
import pyopencl.array
import numpy
import numpy.linalg as la
import collections, math, sys

# should be adjusted according to installation path
sys.path.append('/usr/local/lib/yade-tr2/py')
from miniEigen import *

count=2000

align=16 # alignment for Mat3
N=count*16


# convert flat buffer to miniEigen object, according to type 
def toNumpy(typeName,vec,i):
	if typeName==None: return None
	if typeName=='Vec3': return Vector3(vec[i*4],vec[i*4+1],vec[i*4+2])
	elif typeName=='Mat3': return Matrix3(*tuple([vec[i*16+j] for j in range(0,9)]))
	elif typeName=='double': return float(vec[i])
	elif typeName=='Quat': return Quaternion(vec[i*4+3],vec[i*4],vec[i*4+1],vec[i*4+2])

def orthonorm_c0(m):
	x=m.col(0); y=m.col(1)
	x.normalize()
	y-=x*x.dot(y)
	y.normalize()
	return Matrix3(x,y,x.cross(y),cols=True)
def rot_setYZ(locX):
	locY=Vector3.UnitY if abs(locX[1])<abs(locX[2]) else Vector3.UnitZ
	locY-=locX*locX.dot(locY)
	return Matrix3(locX,locY,locX.cross(locY),cols=True)


a=numpy.random.rand(N).astype(numpy.float64)
b=numpy.random.rand(N).astype(numpy.float64)

ctx=cl.create_some_context(interactive=True)
queue=cl.CommandQueue(ctx)
mf=cl.mem_flags

aBuf=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=a)
bBuf=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=b)
cBuf=cl.Buffer(ctx,mf.WRITE_ONLY,b.nbytes)

CLTest=collections.namedtuple('CLTest',['name','outType','inTypes','clCode','numpyFunc'])


# name: human-readable name
# outType: result type (passed as the 'c' array to the kernel)
# inTypes: input types (passed as the 'a' and 'b' arrays to the kernel)
# clCode: code for the kernel; defined vars are a,b,c, and gid;
# numpyFunc: lambda to be called on inputs to get numpy answer for comparison

tests=[
	# vector
	CLTest('Vec3_unitX','Vec3',(None,None),'c=Vec3_unitX();',lambda a,b: Vector3.UnitX),
	CLTest('Vec3_unitY','Vec3',(None,None),'c=Vec3_unitY();',lambda a,b: Vector3.UnitY),
	CLTest('Vec3_unitZ','Vec3',(None,None),'c=Vec3_unitZ();',lambda a,b: Vector3.UnitZ),
	CLTest('Vec3_zero','Vec3',(None,None),'c=Vec3_zero();',lambda a,b: Vector3.Zero),
	CLTest('Vec3+Vec3','Vec3',('Vec3','Vec3'),'c=a+b;',lambda a,b: a+b),
	CLTest('Vec3*double','Vec3',('Vec3','double'),'c=a*b;',lambda a,b: a*b),
	# matrix built-ins
	CLTest('Mat3+Mat3','Mat3',('Mat3','Mat3'),'c=a+b',lambda a,b: a+b),
	CLTest('Mat3=Mat3','Mat3',('Mat3','Mat3'),'c=a',lambda a,b: a),
	CLTest('double*Mat3','Mat3',('double','Mat3'), 'c=a*b;',lambda a,b: a*b),
	CLTest('Mat3*double','Mat3',('Mat3','double'), 'c=a*b;',lambda a,b: a*b),
	# matrix trivial
	CLTest('Mat3_setRows','Mat3',(None,None),'c=Mat3_setRows((Vec3)0,(Vec3)(1),(Vec3)2);',lambda a,b: Matrix3((0,0,0),(1,1,1),(2,2,2),cols=False)),
	CLTest('Mat3_setCols','Mat3',(None,None),'c=Mat3_setCols((Vec3)0,(Vec3)(1),(Vec3)2);',lambda a,b: Matrix3((0,0,0),(1,1,1),(2,2,2),cols=True)),
	CLTest('Mat3_set','Mat3',(None,None),'c=Mat3_set(0,0.1,0.2,1.0,1.1,1.2,2.0,2.1,2.2);',lambda a,b: Matrix3(0,.1,.2,1.,1.1,1.2,2.,2.1,2.2)),
	# matrix non-trivial
	CLTest('Mat3_transpose','Mat3',('Mat3',None),'c=Mat3_transpose(a)',lambda a,b: a.transpose()),
	CLTest('Vec3_outer','Mat3',('Vec3','Vec3'),  'c=Vec3_outer(a,b)',lambda a,b: a.outer(b)),
	CLTest('Mat3_multM','Mat3',('Mat3','Mat3'), 'c=Mat3_multM(a,b)',lambda a,b: a*b),
	CLTest('Mat3_multV','Vec3',('Mat3','Vec3'), 'c=Mat3_multV(a,b)',lambda a,b: a*b),
	CLTest('Mat3_orthonorm_c0','Mat3',('Mat3',None),'c=Mat3_orthonorm_c0(a)',lambda a,b: orthonorm_c0(a)),
	CLTest('Mat3_rot_setYZ','Mat3',('Vec3',None),'c=Mat3_rot_setYZ(a)',lambda a,b: rot_setYZ(a)),
	CLTest('Mat3_toQuat','Quat',('Mat3',None),'c=Mat3_toQuat(Mat3_orthonorm_c0(a))',lambda a,b: Quaternion(orthonorm_c0(a))),
	CLTest('Quat_toMat3','Mat3',('Quat',None),'c=Quat_toMat3(normalize(a))', lambda a,b: a.normalized().toRotationMatrix()),
	CLTest('Quat_rotate','Vec3',('Quat','Vec3'),'c=Quat_rotate(normalize(a),b);', lambda a,b: a.normalized().Rotate(b)),
	CLTest('Quat_multQ','Quat',('Quat','Quat'),'c=Quat_multQ(normalize(a),normalize(b));', lambda a,b: a.normalized()*b.normalized()),
	#
	CLTest('Mat3_det','double',('Mat3',None),'c=Mat3_det(a)',lambda a,b: a.determinant()),
	CLTest('Mat3_inv','Mat3',('Mat3',None),'c=Mat3_inv(a)',lambda a,b: a.inverse()),
	#
	CLTest('Quat_fromAngleAxis','Quat',('double','Vec3'),'c=Quat_fromAngleAxis(a,normalize(b))',lambda a,b: Quaternion(a,b.normalized())),
	CLTest('Quat_toAngleAxis|angle','double',('Quat',None),'Vec3 foo; Quat_toAngleAxis(normalize(a),&c,&foo)',lambda a,b: a.normalized().toAngleAxis()[0]),
	CLTest('Quat_toAngleAxis|axis','Vec3',('Quat',None),'double foo; Quat_toAngleAxis(normalize(a),&foo,&c)',lambda a,b: a.normalized().toAngleAxis()[1]),
]

#tests=tests[-1:]
dotStride=100

for test in tests:
	print 20*'=',test.name,20*'='
	src='''
		#include<basic-math.cl>
		kernel void test(global {typeC} *cc, global const {typeA} *aa, global const {typeB} /**/ *bb){{ int gid=get_global_id(0); {typeA} __attribute__((unused)) a=aa[gid]; {typeB} __attribute__((unused)) b=bb[gid]; {typeC} c; {clCode}; cc[gid]=c; }}
	'''.format(typeA=(test.inTypes[0] if test.inTypes[0] else 'double'),typeB=(test.inTypes[1] if test.inTypes[1] else 'double'),typeC=test.outType,clCode=test.clCode)	
	prg=cl.Program(ctx,src).build(options="-I.")
	prg.test(queue,(a.shape[0]/align,),None,cBuf,aBuf,bBuf) # we pass uselessly aBuf, bBuf even if they are unused
	c=numpy.empty_like(a) # result will be copied in here
	cl.enqueue_read_buffer(queue,cBuf,c).wait() # compute, wait and copy back
	# go through all results, compare with numpy
	for i in range(0,N/align):
		A,B=toNumpy(test.inTypes[0],a,i),toNumpy(test.inTypes[1],b,i)
		C=toNumpy(test.outType,c,i)
		D=test.numpyFunc(A,B) # call numpy on the args
		# all elements exactly the same
		if C==D:
			if i%dotStride==0: sys.stdout.write('.')
		else:
			if C.__class__==float: relErr=(C-D)/C
			else: relErr=((C-D).norm()/C.norm())
			if relErr<3e-14:
				if i%dotStride==0: sys.stdout.write(':')
			elif relErr<1e-8: sys.stdout.write('[%g]'%relErr)
			else:
				print 20*'@','error (relative %g)'%relErr
				if test.inTypes[0]: print 'input A:',A
				if test.inTypes[1]: print 'input B:',B
				#print 'CL raw:\n',c
				print 'CL     :',C
				print 'numpy  :',D
	print 
		

		
	

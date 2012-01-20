import pyopencl as cl
import pyopencl.array
import numpy
import numpy.linalg as la
import collections
import sys

count=2
align=16 # alignment for Mat3
N=count*16


# convert flat buffer to numpy tupe, according to type 
def toNumpy(typeName,vec,i):
	if typeName==None: return None
	if typeName=='Vec3': return numpy.matrix(vec[i*4:i*4+3].reshape((3,1)))
	elif typeName=='Mat3': return numpy.matrix(vec[i*16:i*16+9].reshape((3,3)))
	elif typeName=='double': return float(vec[i])
	elif typeName=='Quat': return Quaternion(vec[i*4+3],vec[i*4],vec[i*4+1],vec[i*4+2])

def orthonorm_c0(m):
	x=m[:,0]; y=m[:,1]
	x/=la.norm(x)
	y-=x*numpy.dot(x.T,y)
	y/=la.norm(y)
	m[:,0]=x; m[:,1]=y; m[:,2]=numpy.cross(x.T,y.T).T
	return m


a=numpy.random.rand(N).astype(numpy.float64)
b=numpy.random.rand(N).astype(numpy.float64)
aRot=numpy.zeros(N).astype(numpy.float64)
for i in range(0,count):
	m=numpy.matrix(numpy.random.random((3,3)))
	m=orthonorm_c0(m)
	aRot[align*i:align*i+9]=m.reshape((1,9))

ctx=cl.create_some_context(interactive=False)
queue=cl.CommandQueue(ctx)
mf=cl.mem_flags

aBuf=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=a)
bBuf=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=b)
aRotBuf=cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=aRot)
cBuf=cl.Buffer(ctx,mf.WRITE_ONLY,b.nbytes)

CLTest=collections.namedtuple('CLTest',['name','outType','inTypes','clCode','numpyFunc'])


# name: human-readable name
# outType: result type (passed as the 'c' array to the kernel)
# inTypes: input types (passed as the 'a' and 'b' arrays to the kernel)
# clCode: code for the kernel; defined vars are a,b,c, and gid;
# numpyFunc: lambda to be called on inputs to get numpy answer for comparison

tests=[
	# vector
	CLTest('Vec3_unitX','Vec3',(None,None),'c=Vec3_unitX();',lambda a,b: numpy.matrix([[1,0,0]]).T),
	CLTest('Vec3_unitY','Vec3',(None,None),'c=Vec3_unitY();',lambda a,b: numpy.matrix([[0,1,0]]).T),
	CLTest('Vec3_unitZ','Vec3',(None,None),'c=Vec3_unitZ();',lambda a,b: numpy.matrix([[0,0,1]]).T),
	CLTest('Vec3_zero','Vec3',(None,None),'c=Vec3_zero();',lambda a,b: numpy.matrix([[0,0,0]]).T),
	CLTest('Vec3+Vec3','Vec3',('Vec3','Vec3'),'c=a+b;',lambda a,b: numpy.array([a])+numpy.array([b])),
	CLTest('Vec3*double','Vec3',('Vec3','double'),'c=a*b;',lambda a,b: numpy.array([a])*b),
	# matrix built-ins
	CLTest('Mat3+Mat3','Mat3',('Mat3','Mat3'),'c=a+b',lambda a,b: a+b),
	CLTest('Mat3=Mat3','Mat3',('Mat3','Mat3'),'c=a',lambda a,b: a),
	CLTest('double*Mat3','Mat3',('double','Mat3'), 'c=a*b;',lambda a,b: a*b),
	CLTest('Mat3*double','Mat3',('Mat3','double'), 'c=a*b;',lambda a,b: a*b),
	# matrix trivial
	CLTest('Mat3_setRows','Mat3',(None,None),'c=Mat3_setRows((Vec3)0,(Vec3)(1),(Vec3)2);',lambda a,b: numpy.matrix([[0,0,0],[1,1,1],[2,2,2]])),
	CLTest('Mat3_setCols','Mat3',(None,None),'c=Mat3_setCols((Vec3)0,(Vec3)(1),(Vec3)2);',lambda a,b: numpy.matrix([[0,0,0],[1,1,1],[2,2,2]]).T),
	CLTest('Mat3_set','Mat3',(None,None),'c=Mat3_set(0,0.1,0.2,1.0,1.1,1.2,2.0,2.1,2.2);',lambda a,b: numpy.matrix([[0,.1,.2],[1.,1.1,1.2],[2.,2.1,2.2]])),
	# matrix non-trivial
	CLTest('Mat3_transpose','Mat3',('Mat3',None),'c=Mat3_transpose(a)',lambda a,b: a.T),
	CLTest('Vec3_outer','Mat3',('Vec3','Vec3'),  'c=Vec3_outer(a,b)',lambda a,b: numpy.outer(a,b)),
	CLTest('Mat3_multM','Mat3',('Mat3','Mat3'), 'c=Mat3_multM(a,b)',lambda a,b: a*b),
	CLTest('Mat3_multV','Vec3',('Mat3','Vec3'), 'c=Mat3_multV(a,b)',lambda a,b: a*b),
	CLTest('Mat3_orthonorm_c0','Mat3',('Mat3',None),'c=Mat3_orthonorm_c0(a)',lambda a,b: orthonorm_c0(a)),
	#CLTest('Mat3_toQuat [rot]','Quat',('Mat3',None),'c=Mat3_toQuat(a)',lambda a,b: Quaternion(Matrix3(a)))
]

#tests=[tests[-1]]





for test in tests:
	print 20*'=',test.name,20*'='
	rotMatAsInput='[rot]' in test.name
	src='''
		#include<basic-math.cl>
		kernel void test(global {typeC} *cc, global const {typeA} *aa, global const {typeB} /**/ *bb){{ int gid=get_global_id(0); {typeA} a=aa[gid]; {typeB} b=bb[gid]; {typeC} c; {clCode}; cc[gid]=c; }}
	'''.format(typeA=(test.inTypes[0] if test.inTypes[0] else 'double'),typeB=(test.inTypes[1] if test.inTypes[1] else 'double'),typeC=test.outType,clCode=test.clCode)	
	prg=cl.Program(ctx,src).build(options="-I.")
	prg.test(queue,(a.shape[0]/align,),None,cBuf,aBuf if not rotMatAsInput else aRotBuf,bBuf) # we pass uselessly aBuf, bBuf even if they are unused
	c=numpy.empty_like(a) # result will be copied in here
	cl.enqueue_read_buffer(queue,cBuf,c).wait() # compute, wait and copy back
	# go through all results, compare with numpy
	for i in range(0,N/align):
		A,B=toNumpy(test.inTypes[0],a if not rotMatAsInput else aRot,i),toNumpy(test.inTypes[1],b,i)
		C=toNumpy(test.outType,c,i)
		D=test.numpyFunc(A,B) # call numpy on the args
		if numpy.all(C==D):   # all elements exactly the same
			sys.stdout.write('.')
		else:
			relErr=la.norm(C-D)/la.norm(C)
			if relErr<1e-14:
				sys.stdout.write(':')
			elif relErr<1e-8:
				sys.stdout.write('[%g]'%relErr)
			else:
				print 20*'@','error'
				if test.inTypes[0]: print 'input A:\n',A
				if test.inTypes[1]: print 'input B:\n',B
				#print 'CL raw:\n',c
				print 'CL:\n',C
				print 'numpy:\n',D
	print 
		

		
	

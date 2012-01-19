import pyopencl as cl
import pyopencl.array
import numpy
import numpy.linalg as la
import collections
import sys

count=200
align=16 # alignment for Mat33
N=count*16

a=numpy.random.rand(N).astype(numpy.float64)
b=numpy.random.rand(N).astype(numpy.float64)

ctx=cl.create_some_context(interactive=False)
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
	# vector built-ins
	# matrix built-ins
	CLTest('matrix addition','Mat33',('Mat33','Mat33'),'c[gid]=a[gid]+b[gid]',lambda a,b: a+b),
	CLTest('matrix identity','Mat33',('Mat33','Mat33'),'c[gid]=a[gid]',lambda a,b: a),
	CLTest('scalar-matrix product','Mat33',('double','Mat33'), 'c[gid]=a[gid]*b[gid];',lambda a,b: a*b),
	CLTest('matrix-scalar product','Mat33',('Mat33','double'), 'c[gid]=a[gid]*b[gid];',lambda a,b: a*b),
	# matrix trivial
	CLTest('set rows','Mat33',(None,None),'c[gid]=m33setRows((Vec3)0,(Vec3)(1),(Vec3)2);',lambda a,b: numpy.matrix([[0,0,0],[1,1,1],[2,2,2]])),
	CLTest('set cols','Mat33',(None,None),'c[gid]=m33setCols((Vec3)0,(Vec3)(1),(Vec3)2);',lambda a,b: numpy.matrix([[0,0,0],[1,1,1],[2,2,2]]).T),
	CLTest('set components','Mat33',(None,None),'c[gid]=m33set(0,0.1,0.2,1.0,1.1,1.2,2.0,2.1,2.2);',lambda a,b: numpy.matrix([[0,.1,.2],[1.,1.1,1.2],[2.,2.1,2.2]])),
	# matrix non-trivial
	CLTest('transpose','Mat33',('Mat33',None),'c[gid]=m33trans(a[gid])',lambda a,b: a.T),
	CLTest('outer product','Mat33',('Vec3','Vec3'),  'c[gid]=outerVV3(a[gid],b[gid])',lambda a,b: numpy.outer(a,b)),
	CLTest('matrix-matrix product','Mat33',('Mat33','Mat33'), 'c[gid]=mulMM3(a[gid],b[gid])',lambda a,b: a*b),
	CLTest('matrix-vector product','Vec3',('Mat33','Vec3'), 'c[gid]=mulMV3(a[gid],b[gid])',lambda a,b: a*b),
]

# convert flat buffer to numpy tupe, according to type 
def toNumpy(typeName,vec,i):
	if typeName==None: return None
	if typeName=='Vec3': return numpy.matrix(vec[i*4:i*4+3].reshape((3,1)))
	elif typeName=='Mat33': return numpy.matrix(vec[i*16:i*16+9].reshape((3,3)))
	elif typeName=='double': return float(vec[i])

for test in tests:
	print 20*'=',test.name,20*'='
	prg=cl.Program(ctx,'''
		#include"basic-math.cl"
		kernel void test(global %s *c, global const %s *a, global const %s *b){ int gid=get_global_id(0); %s; }
	'''%(test.outType,test.inTypes[0] if test.inTypes[0] else 'Vec3',test.inTypes[1] if test.inTypes[1] else 'Vec3',test.clCode)).build()
	prg.test(queue,(a.shape[0]/align,),None,cBuf,aBuf,bBuf) # we pass uselessly aBuf, bBuf even if they are unused
	c=numpy.empty_like(a) # result will be copied in here
	cl.enqueue_read_buffer(queue,cBuf,c).wait() # compute, wait and copy back
	# go through all results, compare with numpy
	for i in range(0,N/align):
		A,B=toNumpy(test.inTypes[0],a,i),toNumpy(test.inTypes[1],b,i)
		C=toNumpy(test.outType,c,i)
		D=test.numpyFunc(A,B) # call numpy on the args
		if numpy.all(C==D):   # all elements exactly the same
			sys.stdout.write('.')
		else:
			print 20*'@','error'
			if test.inTypes[0]: print 'input A:\n',A
			if test.inTypes[1]: print 'input B:\n',B
			print 'CL raw:\n',c
			print 'CL:\n',C
			print 'numpy:\n',D
	print 
		

		
	

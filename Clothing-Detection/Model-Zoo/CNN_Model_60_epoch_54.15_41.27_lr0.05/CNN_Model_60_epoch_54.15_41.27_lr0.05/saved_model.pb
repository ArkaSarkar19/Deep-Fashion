??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:@*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:@*
dtype0
t
bn_conv1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebn_conv1/gamma
m
"bn_conv1/gamma/Read/ReadVariableOpReadVariableOpbn_conv1/gamma*
_output_shapes
:@*
dtype0
r
bn_conv1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebn_conv1/beta
k
!bn_conv1/beta/Read/ReadVariableOpReadVariableOpbn_conv1/beta*
_output_shapes
:@*
dtype0
?
bn_conv1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namebn_conv1/moving_mean
y
(bn_conv1/moving_mean/Read/ReadVariableOpReadVariableOpbn_conv1/moving_mean*
_output_shapes
:@*
dtype0
?
bn_conv1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebn_conv1/moving_variance
?
,bn_conv1/moving_variance/Read/ReadVariableOpReadVariableOpbn_conv1/moving_variance*
_output_shapes
:@*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:@@*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:@*
dtype0
t
bn_conv2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebn_conv2/gamma
m
"bn_conv2/gamma/Read/ReadVariableOpReadVariableOpbn_conv2/gamma*
_output_shapes
:@*
dtype0
r
bn_conv2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebn_conv2/beta
k
!bn_conv2/beta/Read/ReadVariableOpReadVariableOpbn_conv2/beta*
_output_shapes
:@*
dtype0
?
bn_conv2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namebn_conv2/moving_mean
y
(bn_conv2/moving_mean/Read/ReadVariableOpReadVariableOpbn_conv2/moving_mean*
_output_shapes
:@*
dtype0
?
bn_conv2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebn_conv2/moving_variance
?
,bn_conv2/moving_variance/Read/ReadVariableOpReadVariableOpbn_conv2/moving_variance*
_output_shapes
:@*
dtype0
}
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameconv3/kernel
v
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*'
_output_shapes
:@?*
dtype0
m

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv3/bias
f
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes	
:?*
dtype0
u
bn_conv3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebn_conv3/gamma
n
"bn_conv3/gamma/Read/ReadVariableOpReadVariableOpbn_conv3/gamma*
_output_shapes	
:?*
dtype0
s
bn_conv3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebn_conv3/beta
l
!bn_conv3/beta/Read/ReadVariableOpReadVariableOpbn_conv3/beta*
_output_shapes	
:?*
dtype0
?
bn_conv3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_conv3/moving_mean
z
(bn_conv3/moving_mean/Read/ReadVariableOpReadVariableOpbn_conv3/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_conv3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebn_conv3/moving_variance
?
,bn_conv3/moving_variance/Read/ReadVariableOpReadVariableOpbn_conv3/moving_variance*
_output_shapes	
:?*
dtype0
~
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv4/kernel
w
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*(
_output_shapes
:??*
dtype0
m

conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv4/bias
f
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes	
:?*
dtype0
u
bn_conv4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebn_conv4/gamma
n
"bn_conv4/gamma/Read/ReadVariableOpReadVariableOpbn_conv4/gamma*
_output_shapes	
:?*
dtype0
s
bn_conv4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebn_conv4/beta
l
!bn_conv4/beta/Read/ReadVariableOpReadVariableOpbn_conv4/beta*
_output_shapes	
:?*
dtype0
?
bn_conv4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_conv4/moving_mean
z
(bn_conv4/moving_mean/Read/ReadVariableOpReadVariableOpbn_conv4/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_conv4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebn_conv4/moving_variance
?
,bn_conv4/moving_variance/Read/ReadVariableOpReadVariableOpbn_conv4/moving_variance*
_output_shapes	
:?*
dtype0
r

fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*
shared_name
fc1/kernel
k
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel* 
_output_shapes
:
?	?*
dtype0
i
fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
fc1/bias
b
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
_output_shapes	
:?*
dtype0
r

fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name
fc2/kernel
k
fc2/kernel/Read/ReadVariableOpReadVariableOp
fc2/kernel* 
_output_shapes
:
??*
dtype0
i
fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
fc2/bias
b
fc2/bias/Read/ReadVariableOpReadVariableOpfc2/bias*
_output_shapes	
:?*
dtype0
q

fc3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name
fc3/kernel
j
fc3/kernel/Read/ReadVariableOpReadVariableOp
fc3/kernel*
_output_shapes
:	?*
dtype0
h
fc3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc3/bias
a
fc3/bias/Read/ReadVariableOpReadVariableOpfc3/bias*
_output_shapes
:*
dtype0
n
Adadelta/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
_output_shapes
: *
dtype0	
p
Adadelta/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
_output_shapes
: *
dtype0
?
Adadelta/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
_output_shapes
: *
dtype0
l
Adadelta/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/rho
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
 Adadelta/conv1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adadelta/conv1/kernel/accum_grad
?
4Adadelta/conv1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/conv1/kernel/accum_grad*&
_output_shapes
:@*
dtype0
?
Adadelta/conv1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adadelta/conv1/bias/accum_grad
?
2Adadelta/conv1/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/conv1/bias/accum_grad*
_output_shapes
:@*
dtype0
?
"Adadelta/bn_conv1/gamma/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adadelta/bn_conv1/gamma/accum_grad
?
6Adadelta/bn_conv1/gamma/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/bn_conv1/gamma/accum_grad*
_output_shapes
:@*
dtype0
?
!Adadelta/bn_conv1/beta/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adadelta/bn_conv1/beta/accum_grad
?
5Adadelta/bn_conv1/beta/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv1/beta/accum_grad*
_output_shapes
:@*
dtype0
?
 Adadelta/conv2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*1
shared_name" Adadelta/conv2/kernel/accum_grad
?
4Adadelta/conv2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/conv2/kernel/accum_grad*&
_output_shapes
:@@*
dtype0
?
Adadelta/conv2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adadelta/conv2/bias/accum_grad
?
2Adadelta/conv2/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/conv2/bias/accum_grad*
_output_shapes
:@*
dtype0
?
"Adadelta/bn_conv2/gamma/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adadelta/bn_conv2/gamma/accum_grad
?
6Adadelta/bn_conv2/gamma/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/bn_conv2/gamma/accum_grad*
_output_shapes
:@*
dtype0
?
!Adadelta/bn_conv2/beta/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adadelta/bn_conv2/beta/accum_grad
?
5Adadelta/bn_conv2/beta/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv2/beta/accum_grad*
_output_shapes
:@*
dtype0
?
 Adadelta/conv3/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*1
shared_name" Adadelta/conv3/kernel/accum_grad
?
4Adadelta/conv3/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/conv3/kernel/accum_grad*'
_output_shapes
:@?*
dtype0
?
Adadelta/conv3/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adadelta/conv3/bias/accum_grad
?
2Adadelta/conv3/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/conv3/bias/accum_grad*
_output_shapes	
:?*
dtype0
?
"Adadelta/bn_conv3/gamma/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adadelta/bn_conv3/gamma/accum_grad
?
6Adadelta/bn_conv3/gamma/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/bn_conv3/gamma/accum_grad*
_output_shapes	
:?*
dtype0
?
!Adadelta/bn_conv3/beta/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adadelta/bn_conv3/beta/accum_grad
?
5Adadelta/bn_conv3/beta/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv3/beta/accum_grad*
_output_shapes	
:?*
dtype0
?
 Adadelta/conv4/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*1
shared_name" Adadelta/conv4/kernel/accum_grad
?
4Adadelta/conv4/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/conv4/kernel/accum_grad*(
_output_shapes
:??*
dtype0
?
Adadelta/conv4/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adadelta/conv4/bias/accum_grad
?
2Adadelta/conv4/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/conv4/bias/accum_grad*
_output_shapes	
:?*
dtype0
?
"Adadelta/bn_conv4/gamma/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adadelta/bn_conv4/gamma/accum_grad
?
6Adadelta/bn_conv4/gamma/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/bn_conv4/gamma/accum_grad*
_output_shapes	
:?*
dtype0
?
!Adadelta/bn_conv4/beta/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adadelta/bn_conv4/beta/accum_grad
?
5Adadelta/bn_conv4/beta/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv4/beta/accum_grad*
_output_shapes	
:?*
dtype0
?
Adadelta/fc1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*/
shared_name Adadelta/fc1/kernel/accum_grad
?
2Adadelta/fc1/kernel/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/fc1/kernel/accum_grad* 
_output_shapes
:
?	?*
dtype0
?
Adadelta/fc1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdadelta/fc1/bias/accum_grad
?
0Adadelta/fc1/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/fc1/bias/accum_grad*
_output_shapes	
:?*
dtype0
?
Adadelta/fc2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adadelta/fc2/kernel/accum_grad
?
2Adadelta/fc2/kernel/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/fc2/kernel/accum_grad* 
_output_shapes
:
??*
dtype0
?
Adadelta/fc2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdadelta/fc2/bias/accum_grad
?
0Adadelta/fc2/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/fc2/bias/accum_grad*
_output_shapes	
:?*
dtype0
?
Adadelta/fc3/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adadelta/fc3/kernel/accum_grad
?
2Adadelta/fc3/kernel/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/fc3/kernel/accum_grad*
_output_shapes
:	?*
dtype0
?
Adadelta/fc3/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdadelta/fc3/bias/accum_grad
?
0Adadelta/fc3/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/fc3/bias/accum_grad*
_output_shapes
:*
dtype0
?
Adadelta/conv1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adadelta/conv1/kernel/accum_var
?
3Adadelta/conv1/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv1/kernel/accum_var*&
_output_shapes
:@*
dtype0
?
Adadelta/conv1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdadelta/conv1/bias/accum_var
?
1Adadelta/conv1/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv1/bias/accum_var*
_output_shapes
:@*
dtype0
?
!Adadelta/bn_conv1/gamma/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adadelta/bn_conv1/gamma/accum_var
?
5Adadelta/bn_conv1/gamma/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv1/gamma/accum_var*
_output_shapes
:@*
dtype0
?
 Adadelta/bn_conv1/beta/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adadelta/bn_conv1/beta/accum_var
?
4Adadelta/bn_conv1/beta/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/bn_conv1/beta/accum_var*
_output_shapes
:@*
dtype0
?
Adadelta/conv2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!Adadelta/conv2/kernel/accum_var
?
3Adadelta/conv2/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv2/kernel/accum_var*&
_output_shapes
:@@*
dtype0
?
Adadelta/conv2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdadelta/conv2/bias/accum_var
?
1Adadelta/conv2/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv2/bias/accum_var*
_output_shapes
:@*
dtype0
?
!Adadelta/bn_conv2/gamma/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adadelta/bn_conv2/gamma/accum_var
?
5Adadelta/bn_conv2/gamma/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv2/gamma/accum_var*
_output_shapes
:@*
dtype0
?
 Adadelta/bn_conv2/beta/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adadelta/bn_conv2/beta/accum_var
?
4Adadelta/bn_conv2/beta/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/bn_conv2/beta/accum_var*
_output_shapes
:@*
dtype0
?
Adadelta/conv3/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*0
shared_name!Adadelta/conv3/kernel/accum_var
?
3Adadelta/conv3/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv3/kernel/accum_var*'
_output_shapes
:@?*
dtype0
?
Adadelta/conv3/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameAdadelta/conv3/bias/accum_var
?
1Adadelta/conv3/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv3/bias/accum_var*
_output_shapes	
:?*
dtype0
?
!Adadelta/bn_conv3/gamma/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adadelta/bn_conv3/gamma/accum_var
?
5Adadelta/bn_conv3/gamma/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv3/gamma/accum_var*
_output_shapes	
:?*
dtype0
?
 Adadelta/bn_conv3/beta/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adadelta/bn_conv3/beta/accum_var
?
4Adadelta/bn_conv3/beta/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/bn_conv3/beta/accum_var*
_output_shapes	
:?*
dtype0
?
Adadelta/conv4/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*0
shared_name!Adadelta/conv4/kernel/accum_var
?
3Adadelta/conv4/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv4/kernel/accum_var*(
_output_shapes
:??*
dtype0
?
Adadelta/conv4/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameAdadelta/conv4/bias/accum_var
?
1Adadelta/conv4/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv4/bias/accum_var*
_output_shapes	
:?*
dtype0
?
!Adadelta/bn_conv4/gamma/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adadelta/bn_conv4/gamma/accum_var
?
5Adadelta/bn_conv4/gamma/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv4/gamma/accum_var*
_output_shapes	
:?*
dtype0
?
 Adadelta/bn_conv4/beta/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adadelta/bn_conv4/beta/accum_var
?
4Adadelta/bn_conv4/beta/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/bn_conv4/beta/accum_var*
_output_shapes	
:?*
dtype0
?
Adadelta/fc1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*.
shared_nameAdadelta/fc1/kernel/accum_var
?
1Adadelta/fc1/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/fc1/kernel/accum_var* 
_output_shapes
:
?	?*
dtype0
?
Adadelta/fc1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdadelta/fc1/bias/accum_var
?
/Adadelta/fc1/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/fc1/bias/accum_var*
_output_shapes	
:?*
dtype0
?
Adadelta/fc2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdadelta/fc2/kernel/accum_var
?
1Adadelta/fc2/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/fc2/kernel/accum_var* 
_output_shapes
:
??*
dtype0
?
Adadelta/fc2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameAdadelta/fc2/bias/accum_var
?
/Adadelta/fc2/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/fc2/bias/accum_var*
_output_shapes	
:?*
dtype0
?
Adadelta/fc3/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameAdadelta/fc3/kernel/accum_var
?
1Adadelta/fc3/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/fc3/kernel/accum_var*
_output_shapes
:	?*
dtype0
?
Adadelta/fc3/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdadelta/fc3/bias/accum_var
?
/Adadelta/fc3/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/fc3/bias/accum_var*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*֙
value˙BǙ B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer_with_weights-9
layer-21
layer_with_weights-10
layer-22
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
?
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
R
1trainable_variables
2	variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
?
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
R
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
R
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
R
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
h

Pkernel
Qbias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
?
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[trainable_variables
\	variables
]regularization_losses
^	keras_api
R
_trainable_variables
`	variables
aregularization_losses
b	keras_api
R
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
R
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
h

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
?
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
T
~trainable_variables
	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?iter

?decay
?learning_rate
?rho
accum_grad?
accum_grad?%
accum_grad?&
accum_grad?5
accum_grad?6
accum_grad?<
accum_grad?=
accum_grad?P
accum_grad?Q
accum_grad?W
accum_grad?X
accum_grad?k
accum_grad?l
accum_grad?r
accum_grad?s
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad?	accum_var?	accum_var?%	accum_var?&	accum_var?5	accum_var?6	accum_var?<	accum_var?=	accum_var?P	accum_var?Q	accum_var?W	accum_var?X	accum_var?k	accum_var?l	accum_var?r	accum_var?s	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var?
?
0
1
%2
&3
54
65
<6
=7
P8
Q9
W10
X11
k12
l13
r14
s15
?16
?17
?18
?19
?20
?21
?
0
1
%2
&3
'4
(5
56
67
<8
=9
>10
?11
P12
Q13
W14
X15
Y16
Z17
k18
l19
r20
s21
t22
u23
?24
?25
?26
?27
?28
?29
 
?
trainable_variables
?layers
 ?layer_regularization_losses
?metrics
	variables
?non_trainable_variables
regularization_losses
?layer_metrics
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
 trainable_variables
 ?layer_regularization_losses
?layers
?metrics
!	variables
?non_trainable_variables
"regularization_losses
?layer_metrics
 
YW
VARIABLE_VALUEbn_conv1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbn_conv1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbn_conv1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn_conv1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
'2
(3
 
?
)trainable_variables
 ?layer_regularization_losses
?layers
?metrics
*	variables
?non_trainable_variables
+regularization_losses
?layer_metrics
 
 
 
?
-trainable_variables
 ?layer_regularization_losses
?layers
?metrics
.	variables
?non_trainable_variables
/regularization_losses
?layer_metrics
 
 
 
?
1trainable_variables
 ?layer_regularization_losses
?layers
?metrics
2	variables
?non_trainable_variables
3regularization_losses
?layer_metrics
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
?
7trainable_variables
 ?layer_regularization_losses
?layers
?metrics
8	variables
?non_trainable_variables
9regularization_losses
?layer_metrics
 
YW
VARIABLE_VALUEbn_conv2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbn_conv2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbn_conv2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn_conv2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
>2
?3
 
?
@trainable_variables
 ?layer_regularization_losses
?layers
?metrics
A	variables
?non_trainable_variables
Bregularization_losses
?layer_metrics
 
 
 
?
Dtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
E	variables
?non_trainable_variables
Fregularization_losses
?layer_metrics
 
 
 
?
Htrainable_variables
 ?layer_regularization_losses
?layers
?metrics
I	variables
?non_trainable_variables
Jregularization_losses
?layer_metrics
 
 
 
?
Ltrainable_variables
 ?layer_regularization_losses
?layers
?metrics
M	variables
?non_trainable_variables
Nregularization_losses
?layer_metrics
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

P0
Q1
 
?
Rtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
S	variables
?non_trainable_variables
Tregularization_losses
?layer_metrics
 
YW
VARIABLE_VALUEbn_conv3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbn_conv3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbn_conv3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn_conv3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

W0
X1

W0
X1
Y2
Z3
 
?
[trainable_variables
 ?layer_regularization_losses
?layers
?metrics
\	variables
?non_trainable_variables
]regularization_losses
?layer_metrics
 
 
 
?
_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
`	variables
?non_trainable_variables
aregularization_losses
?layer_metrics
 
 
 
?
ctrainable_variables
 ?layer_regularization_losses
?layers
?metrics
d	variables
?non_trainable_variables
eregularization_losses
?layer_metrics
 
 
 
?
gtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
h	variables
?non_trainable_variables
iregularization_losses
?layer_metrics
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

k0
l1
 
?
mtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
n	variables
?non_trainable_variables
oregularization_losses
?layer_metrics
 
YW
VARIABLE_VALUEbn_conv4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbn_conv4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbn_conv4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn_conv4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

r0
s1
t2
u3
 
?
vtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
w	variables
?non_trainable_variables
xregularization_losses
?layer_metrics
 
 
 
?
ztrainable_variables
 ?layer_regularization_losses
?layers
?metrics
{	variables
?non_trainable_variables
|regularization_losses
?layer_metrics
 
 
 
?
~trainable_variables
 ?layer_regularization_losses
?layers
?metrics
	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
VT
VARIABLE_VALUE
fc1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEfc1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
 
 
 
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
VT
VARIABLE_VALUE
fc2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEfc2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
WU
VARIABLE_VALUE
fc3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
 

?0
?1
8
'0
(1
>2
?3
Y4
Z5
t6
u7
 
 
 
 
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

>0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Y0
Z1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

t0
u1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE Adadelta/conv1/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv1/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/bn_conv1/gamma/accum_gradZlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv1/beta/accum_gradYlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/conv2/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv2/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/bn_conv2/gamma/accum_gradZlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv2/beta/accum_gradYlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/conv3/kernel/accum_grad[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv3/bias/accum_gradYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/bn_conv3/gamma/accum_gradZlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv3/beta/accum_gradYlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/conv4/kernel/accum_grad[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv4/bias/accum_gradYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/bn_conv4/gamma/accum_gradZlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv4/beta/accum_gradYlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/kernel/accum_grad[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/bias/accum_gradYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/kernel/accum_grad[layer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/bias/accum_gradYlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/kernel/accum_grad\layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/bias/accum_gradZlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv1/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv1/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv1/gamma/accum_varYlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/bn_conv1/beta/accum_varXlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv2/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv2/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv2/gamma/accum_varYlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/bn_conv2/beta/accum_varXlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv3/kernel/accum_varZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv3/bias/accum_varXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv3/gamma/accum_varYlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/bn_conv3/beta/accum_varXlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv4/kernel/accum_varZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv4/bias/accum_varXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv4/gamma/accum_varYlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/bn_conv4/beta/accum_varXlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/kernel/accum_varZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/bias/accum_varXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/kernel/accum_varZlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/bias/accum_varXlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/kernel/accum_var[layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/bias/accum_varYlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1/kernel
conv1/biasbn_conv1/gammabn_conv1/betabn_conv1/moving_meanbn_conv1/moving_varianceconv2/kernel
conv2/biasbn_conv2/gammabn_conv2/betabn_conv2/moving_meanbn_conv2/moving_varianceconv3/kernel
conv3/biasbn_conv3/gammabn_conv3/betabn_conv3/moving_meanbn_conv3/moving_varianceconv4/kernel
conv4/biasbn_conv4/gammabn_conv4/betabn_conv4/moving_meanbn_conv4/moving_variance
fc1/kernelfc1/bias
fc2/kernelfc2/bias
fc3/kernelfc3/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_37583
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp"bn_conv1/gamma/Read/ReadVariableOp!bn_conv1/beta/Read/ReadVariableOp(bn_conv1/moving_mean/Read/ReadVariableOp,bn_conv1/moving_variance/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp"bn_conv2/gamma/Read/ReadVariableOp!bn_conv2/beta/Read/ReadVariableOp(bn_conv2/moving_mean/Read/ReadVariableOp,bn_conv2/moving_variance/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp"bn_conv3/gamma/Read/ReadVariableOp!bn_conv3/beta/Read/ReadVariableOp(bn_conv3/moving_mean/Read/ReadVariableOp,bn_conv3/moving_variance/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp"bn_conv4/gamma/Read/ReadVariableOp!bn_conv4/beta/Read/ReadVariableOp(bn_conv4/moving_mean/Read/ReadVariableOp,bn_conv4/moving_variance/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOpfc2/kernel/Read/ReadVariableOpfc2/bias/Read/ReadVariableOpfc3/kernel/Read/ReadVariableOpfc3/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Adadelta/conv1/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv1/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv1/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv1/beta/accum_grad/Read/ReadVariableOp4Adadelta/conv2/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv2/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv2/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv2/beta/accum_grad/Read/ReadVariableOp4Adadelta/conv3/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv3/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv3/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv3/beta/accum_grad/Read/ReadVariableOp4Adadelta/conv4/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv4/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv4/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv4/beta/accum_grad/Read/ReadVariableOp2Adadelta/fc1/kernel/accum_grad/Read/ReadVariableOp0Adadelta/fc1/bias/accum_grad/Read/ReadVariableOp2Adadelta/fc2/kernel/accum_grad/Read/ReadVariableOp0Adadelta/fc2/bias/accum_grad/Read/ReadVariableOp2Adadelta/fc3/kernel/accum_grad/Read/ReadVariableOp0Adadelta/fc3/bias/accum_grad/Read/ReadVariableOp3Adadelta/conv1/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv1/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv1/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv1/beta/accum_var/Read/ReadVariableOp3Adadelta/conv2/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv2/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv2/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv2/beta/accum_var/Read/ReadVariableOp3Adadelta/conv3/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv3/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv3/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv3/beta/accum_var/Read/ReadVariableOp3Adadelta/conv4/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv4/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv4/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv4/beta/accum_var/Read/ReadVariableOp1Adadelta/fc1/kernel/accum_var/Read/ReadVariableOp/Adadelta/fc1/bias/accum_var/Read/ReadVariableOp1Adadelta/fc2/kernel/accum_var/Read/ReadVariableOp/Adadelta/fc2/bias/accum_var/Read/ReadVariableOp1Adadelta/fc3/kernel/accum_var/Read/ReadVariableOp/Adadelta/fc3/bias/accum_var/Read/ReadVariableOpConst*_
TinX
V2T	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_39025
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasbn_conv1/gammabn_conv1/betabn_conv1/moving_meanbn_conv1/moving_varianceconv2/kernel
conv2/biasbn_conv2/gammabn_conv2/betabn_conv2/moving_meanbn_conv2/moving_varianceconv3/kernel
conv3/biasbn_conv3/gammabn_conv3/betabn_conv3/moving_meanbn_conv3/moving_varianceconv4/kernel
conv4/biasbn_conv4/gammabn_conv4/betabn_conv4/moving_meanbn_conv4/moving_variance
fc1/kernelfc1/bias
fc2/kernelfc2/bias
fc3/kernelfc3/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_1 Adadelta/conv1/kernel/accum_gradAdadelta/conv1/bias/accum_grad"Adadelta/bn_conv1/gamma/accum_grad!Adadelta/bn_conv1/beta/accum_grad Adadelta/conv2/kernel/accum_gradAdadelta/conv2/bias/accum_grad"Adadelta/bn_conv2/gamma/accum_grad!Adadelta/bn_conv2/beta/accum_grad Adadelta/conv3/kernel/accum_gradAdadelta/conv3/bias/accum_grad"Adadelta/bn_conv3/gamma/accum_grad!Adadelta/bn_conv3/beta/accum_grad Adadelta/conv4/kernel/accum_gradAdadelta/conv4/bias/accum_grad"Adadelta/bn_conv4/gamma/accum_grad!Adadelta/bn_conv4/beta/accum_gradAdadelta/fc1/kernel/accum_gradAdadelta/fc1/bias/accum_gradAdadelta/fc2/kernel/accum_gradAdadelta/fc2/bias/accum_gradAdadelta/fc3/kernel/accum_gradAdadelta/fc3/bias/accum_gradAdadelta/conv1/kernel/accum_varAdadelta/conv1/bias/accum_var!Adadelta/bn_conv1/gamma/accum_var Adadelta/bn_conv1/beta/accum_varAdadelta/conv2/kernel/accum_varAdadelta/conv2/bias/accum_var!Adadelta/bn_conv2/gamma/accum_var Adadelta/bn_conv2/beta/accum_varAdadelta/conv3/kernel/accum_varAdadelta/conv3/bias/accum_var!Adadelta/bn_conv3/gamma/accum_var Adadelta/bn_conv3/beta/accum_varAdadelta/conv4/kernel/accum_varAdadelta/conv4/bias/accum_var!Adadelta/bn_conv4/gamma/accum_var Adadelta/bn_conv4/beta/accum_varAdadelta/fc1/kernel/accum_varAdadelta/fc1/bias/accum_varAdadelta/fc2/kernel/accum_varAdadelta/fc2/bias/accum_varAdadelta/fc3/kernel/accum_varAdadelta/fc3/bias/accum_var*^
TinW
U2S*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_39281ֈ
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38015

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_36090

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_36844

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv4_layer_call_fn_38571

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_364382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?[
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37207
input_1
conv1_37124
conv1_37126
bn_conv1_37129
bn_conv1_37131
bn_conv1_37133
bn_conv1_37135
conv2_37140
conv2_37142
bn_conv2_37145
bn_conv2_37147
bn_conv2_37149
bn_conv2_37151
conv3_37157
conv3_37159
bn_conv3_37162
bn_conv3_37164
bn_conv3_37166
bn_conv3_37168
conv4_37174
conv4_37176
bn_conv4_37179
bn_conv4_37181
bn_conv4_37183
bn_conv4_37185
	fc1_37190
	fc1_37192
	fc2_37196
	fc2_37198
	fc3_37201
	fc3_37203
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_37124conv1_37126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_364942
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_37129bn_conv1_37131bn_conv1_37133bn_conv1_37135*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_365472"
 bn_conv1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_365882
activation/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_361382
max_pooling2d/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_37140conv2_37142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_366072
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_37145bn_conv2_37147bn_conv2_37149bn_conv2_37151*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_366602"
 bn_conv2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_367012
activation_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_362542!
max_pooling2d_1/PartitionedCall?
dropout/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_367272
dropout/PartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv3_37157conv3_37159*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_367502
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_37162bn_conv3_37164bn_conv3_37166bn_conv3_37168*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_368032"
 bn_conv3/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_368442
activation_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_363702!
max_pooling2d_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_368702
dropout_1/PartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv4_37174conv4_37176*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_368932
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_37179bn_conv4_37181bn_conv4_37183bn_conv4_37185*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_369462"
 bn_conv4/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_369872
activation_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_370012
flatten/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	fc1_37190	fc1_37192*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_370202
fc1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_370532
dropout_2/PartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0	fc2_37196	fc2_37198*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_370772
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_37201	fc3_37203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc3_layer_call_and_return_conditional_losses_371042
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
K
/__inference_max_pooling2d_2_layer_call_fn_36376

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_363702
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_36237

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_38486

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37729

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource$
 bn_conv1_readvariableop_resource&
"bn_conv1_readvariableop_1_resource5
1bn_conv1_fusedbatchnormv3_readvariableop_resource7
3bn_conv1_fusedbatchnormv3_readvariableop_1_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource$
 bn_conv2_readvariableop_resource&
"bn_conv2_readvariableop_1_resource5
1bn_conv2_fusedbatchnormv3_readvariableop_resource7
3bn_conv2_fusedbatchnormv3_readvariableop_1_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource$
 bn_conv3_readvariableop_resource&
"bn_conv3_readvariableop_1_resource5
1bn_conv3_fusedbatchnormv3_readvariableop_resource7
3bn_conv3_fusedbatchnormv3_readvariableop_1_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource$
 bn_conv4_readvariableop_resource&
"bn_conv4_readvariableop_1_resource5
1bn_conv4_fusedbatchnormv3_readvariableop_resource7
3bn_conv4_fusedbatchnormv3_readvariableop_1_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identity??bn_conv1/AssignNewValue?bn_conv1/AssignNewValue_1?(bn_conv1/FusedBatchNormV3/ReadVariableOp?*bn_conv1/FusedBatchNormV3/ReadVariableOp_1?bn_conv1/ReadVariableOp?bn_conv1/ReadVariableOp_1?bn_conv2/AssignNewValue?bn_conv2/AssignNewValue_1?(bn_conv2/FusedBatchNormV3/ReadVariableOp?*bn_conv2/FusedBatchNormV3/ReadVariableOp_1?bn_conv2/ReadVariableOp?bn_conv2/ReadVariableOp_1?bn_conv3/AssignNewValue?bn_conv3/AssignNewValue_1?(bn_conv3/FusedBatchNormV3/ReadVariableOp?*bn_conv3/FusedBatchNormV3/ReadVariableOp_1?bn_conv3/ReadVariableOp?bn_conv3/ReadVariableOp_1?bn_conv4/AssignNewValue?bn_conv4/AssignNewValue_1?(bn_conv4/FusedBatchNormV3/ReadVariableOp?*bn_conv4/FusedBatchNormV3/ReadVariableOp_1?bn_conv4/ReadVariableOp?bn_conv4/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?fc1/BiasAdd/ReadVariableOp?fc1/MatMul/ReadVariableOp?fc2/BiasAdd/ReadVariableOp?fc2/MatMul/ReadVariableOp?fc3/BiasAdd/ReadVariableOp?fc3/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2
conv1/BiasAdd?
bn_conv1/ReadVariableOpReadVariableOp bn_conv1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_conv1/ReadVariableOp?
bn_conv1/ReadVariableOp_1ReadVariableOp"bn_conv1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_conv1/ReadVariableOp_1?
(bn_conv1/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(bn_conv1/FusedBatchNormV3/ReadVariableOp?
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1?
bn_conv1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn_conv1/ReadVariableOp:value:0!bn_conv1/ReadVariableOp_1:value:00bn_conv1/FusedBatchNormV3/ReadVariableOp:value:02bn_conv1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????<<@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn_conv1/FusedBatchNormV3?
bn_conv1/AssignNewValueAssignVariableOp1bn_conv1_fusedbatchnormv3_readvariableop_resource&bn_conv1/FusedBatchNormV3:batch_mean:0)^bn_conv1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*D
_class:
86loc:@bn_conv1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn_conv1/AssignNewValue?
bn_conv1/AssignNewValue_1AssignVariableOp3bn_conv1_fusedbatchnormv3_readvariableop_1_resource*bn_conv1/FusedBatchNormV3:batch_variance:0+^bn_conv1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*F
_class<
:8loc:@bn_conv1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn_conv1/AssignNewValue_1?
activation/ReluRelubn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????<<@2
activation/Relu?
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2/BiasAdd?
bn_conv2/ReadVariableOpReadVariableOp bn_conv2_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_conv2/ReadVariableOp?
bn_conv2/ReadVariableOp_1ReadVariableOp"bn_conv2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_conv2/ReadVariableOp_1?
(bn_conv2/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(bn_conv2/FusedBatchNormV3/ReadVariableOp?
*bn_conv2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*bn_conv2/FusedBatchNormV3/ReadVariableOp_1?
bn_conv2/FusedBatchNormV3FusedBatchNormV3conv2/BiasAdd:output:0bn_conv2/ReadVariableOp:value:0!bn_conv2/ReadVariableOp_1:value:00bn_conv2/FusedBatchNormV3/ReadVariableOp:value:02bn_conv2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn_conv2/FusedBatchNormV3?
bn_conv2/AssignNewValueAssignVariableOp1bn_conv2_fusedbatchnormv3_readvariableop_resource&bn_conv2/FusedBatchNormV3:batch_mean:0)^bn_conv2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*D
_class:
86loc:@bn_conv2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn_conv2/AssignNewValue?
bn_conv2/AssignNewValue_1AssignVariableOp3bn_conv2_fusedbatchnormv3_readvariableop_1_resource*bn_conv2/FusedBatchNormV3:batch_variance:0+^bn_conv2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*F
_class<
:8loc:@bn_conv2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn_conv2/AssignNewValue_1?
activation_1/ReluRelubn_conv2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMul max_pooling2d_1/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/dropout/Mul~
dropout/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/dropout/Mul_1?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Ddropout/dropout/Mul_1:z:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3/BiasAdd?
bn_conv3/ReadVariableOpReadVariableOp bn_conv3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_conv3/ReadVariableOp?
bn_conv3/ReadVariableOp_1ReadVariableOp"bn_conv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_conv3/ReadVariableOp_1?
(bn_conv3/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(bn_conv3/FusedBatchNormV3/ReadVariableOp?
*bn_conv3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02,
*bn_conv3/FusedBatchNormV3/ReadVariableOp_1?
bn_conv3/FusedBatchNormV3FusedBatchNormV3conv3/BiasAdd:output:0bn_conv3/ReadVariableOp:value:0!bn_conv3/ReadVariableOp_1:value:00bn_conv3/FusedBatchNormV3/ReadVariableOp:value:02bn_conv3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn_conv3/FusedBatchNormV3?
bn_conv3/AssignNewValueAssignVariableOp1bn_conv3_fusedbatchnormv3_readvariableop_resource&bn_conv3/FusedBatchNormV3:batch_mean:0)^bn_conv3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*D
_class:
86loc:@bn_conv3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn_conv3/AssignNewValue?
bn_conv3/AssignNewValue_1AssignVariableOp3bn_conv3_fusedbatchnormv3_readvariableop_1_resource*bn_conv3/FusedBatchNormV3:batch_variance:0+^bn_conv3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*F
_class<
:8loc:@bn_conv3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn_conv3/AssignNewValue_1?
activation_2/ReluRelubn_conv3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul max_pooling2d_2/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_1/dropout/Mul_1?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4/Conv2D/ReadVariableOp?
conv4/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv4/Conv2D?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv4/BiasAdd/ReadVariableOp?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4/BiasAdd?
bn_conv4/ReadVariableOpReadVariableOp bn_conv4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_conv4/ReadVariableOp?
bn_conv4/ReadVariableOp_1ReadVariableOp"bn_conv4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_conv4/ReadVariableOp_1?
(bn_conv4/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(bn_conv4/FusedBatchNormV3/ReadVariableOp?
*bn_conv4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02,
*bn_conv4/FusedBatchNormV3/ReadVariableOp_1?
bn_conv4/FusedBatchNormV3FusedBatchNormV3conv4/BiasAdd:output:0bn_conv4/ReadVariableOp:value:0!bn_conv4/ReadVariableOp_1:value:00bn_conv4/FusedBatchNormV3/ReadVariableOp:value:02bn_conv4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn_conv4/FusedBatchNormV3?
bn_conv4/AssignNewValueAssignVariableOp1bn_conv4_fusedbatchnormv3_readvariableop_resource&bn_conv4/FusedBatchNormV3:batch_mean:0)^bn_conv4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*D
_class:
86loc:@bn_conv4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn_conv4/AssignNewValue?
bn_conv4/AssignNewValue_1AssignVariableOp3bn_conv4_fusedbatchnormv3_readvariableop_1_resource*bn_conv4/FusedBatchNormV3:batch_variance:0+^bn_conv4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*F
_class<
:8loc:@bn_conv4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn_conv4/AssignNewValue_1?
activation_3/ReluRelubn_conv4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeactivation_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
fc1/MatMul/ReadVariableOp?

fc1/MatMulMatMulflatten/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

fc1/MatMul?
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
fc1/BiasAdd/ReadVariableOp?
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

fc1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulfc1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mulx
dropout_2/dropout/ShapeShapefc1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mul_1?
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
fc2/MatMul/ReadVariableOp?

fc2/MatMulMatMuldropout_2/dropout/Mul_1:z:0!fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

fc2/MatMul?
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
fc2/BiasAdd/ReadVariableOp?
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
fc2/BiasAdde
fc2/ReluRelufc2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

fc2/Relu?
fc3/MatMul/ReadVariableOpReadVariableOp"fc3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
fc3/MatMul/ReadVariableOp?

fc3/MatMulMatMulfc2/Relu:activations:0!fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

fc3/MatMul?
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc3/BiasAdd/ReadVariableOp?
fc3/BiasAddBiasAddfc3/MatMul:product:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fc3/BiasAddm
fc3/SoftmaxSoftmaxfc3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
fc3/Softmax?

IdentityIdentityfc3/Softmax:softmax:0^bn_conv1/AssignNewValue^bn_conv1/AssignNewValue_1)^bn_conv1/FusedBatchNormV3/ReadVariableOp+^bn_conv1/FusedBatchNormV3/ReadVariableOp_1^bn_conv1/ReadVariableOp^bn_conv1/ReadVariableOp_1^bn_conv2/AssignNewValue^bn_conv2/AssignNewValue_1)^bn_conv2/FusedBatchNormV3/ReadVariableOp+^bn_conv2/FusedBatchNormV3/ReadVariableOp_1^bn_conv2/ReadVariableOp^bn_conv2/ReadVariableOp_1^bn_conv3/AssignNewValue^bn_conv3/AssignNewValue_1)^bn_conv3/FusedBatchNormV3/ReadVariableOp+^bn_conv3/FusedBatchNormV3/ReadVariableOp_1^bn_conv3/ReadVariableOp^bn_conv3/ReadVariableOp_1^bn_conv4/AssignNewValue^bn_conv4/AssignNewValue_1)^bn_conv4/FusedBatchNormV3/ReadVariableOp+^bn_conv4/FusedBatchNormV3/ReadVariableOp_1^bn_conv4/ReadVariableOp^bn_conv4/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp^fc3/BiasAdd/ReadVariableOp^fc3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::22
bn_conv1/AssignNewValuebn_conv1/AssignNewValue26
bn_conv1/AssignNewValue_1bn_conv1/AssignNewValue_12T
(bn_conv1/FusedBatchNormV3/ReadVariableOp(bn_conv1/FusedBatchNormV3/ReadVariableOp2X
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1*bn_conv1/FusedBatchNormV3/ReadVariableOp_122
bn_conv1/ReadVariableOpbn_conv1/ReadVariableOp26
bn_conv1/ReadVariableOp_1bn_conv1/ReadVariableOp_122
bn_conv2/AssignNewValuebn_conv2/AssignNewValue26
bn_conv2/AssignNewValue_1bn_conv2/AssignNewValue_12T
(bn_conv2/FusedBatchNormV3/ReadVariableOp(bn_conv2/FusedBatchNormV3/ReadVariableOp2X
*bn_conv2/FusedBatchNormV3/ReadVariableOp_1*bn_conv2/FusedBatchNormV3/ReadVariableOp_122
bn_conv2/ReadVariableOpbn_conv2/ReadVariableOp26
bn_conv2/ReadVariableOp_1bn_conv2/ReadVariableOp_122
bn_conv3/AssignNewValuebn_conv3/AssignNewValue26
bn_conv3/AssignNewValue_1bn_conv3/AssignNewValue_12T
(bn_conv3/FusedBatchNormV3/ReadVariableOp(bn_conv3/FusedBatchNormV3/ReadVariableOp2X
*bn_conv3/FusedBatchNormV3/ReadVariableOp_1*bn_conv3/FusedBatchNormV3/ReadVariableOp_122
bn_conv3/ReadVariableOpbn_conv3/ReadVariableOp26
bn_conv3/ReadVariableOp_1bn_conv3/ReadVariableOp_122
bn_conv4/AssignNewValuebn_conv4/AssignNewValue26
bn_conv4/AssignNewValue_1bn_conv4/AssignNewValue_12T
(bn_conv4/FusedBatchNormV3/ReadVariableOp(bn_conv4/FusedBatchNormV3/ReadVariableOp2X
*bn_conv4/FusedBatchNormV3/ReadVariableOp_1*bn_conv4/FusedBatchNormV3/ReadVariableOp_122
bn_conv4/ReadVariableOpbn_conv4/ReadVariableOp26
bn_conv4/ReadVariableOp_1bn_conv4/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp28
fc3/BiasAdd/ReadVariableOpfc3/BiasAdd/ReadVariableOp26
fc3/MatMul/ReadVariableOpfc3/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_37053

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv2_layer_call_fn_38280

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_362372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38558

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?-
!__inference__traced_restore_39281
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias%
!assignvariableop_2_bn_conv1_gamma$
 assignvariableop_3_bn_conv1_beta+
'assignvariableop_4_bn_conv1_moving_mean/
+assignvariableop_5_bn_conv1_moving_variance#
assignvariableop_6_conv2_kernel!
assignvariableop_7_conv2_bias%
!assignvariableop_8_bn_conv2_gamma$
 assignvariableop_9_bn_conv2_beta,
(assignvariableop_10_bn_conv2_moving_mean0
,assignvariableop_11_bn_conv2_moving_variance$
 assignvariableop_12_conv3_kernel"
assignvariableop_13_conv3_bias&
"assignvariableop_14_bn_conv3_gamma%
!assignvariableop_15_bn_conv3_beta,
(assignvariableop_16_bn_conv3_moving_mean0
,assignvariableop_17_bn_conv3_moving_variance$
 assignvariableop_18_conv4_kernel"
assignvariableop_19_conv4_bias&
"assignvariableop_20_bn_conv4_gamma%
!assignvariableop_21_bn_conv4_beta,
(assignvariableop_22_bn_conv4_moving_mean0
,assignvariableop_23_bn_conv4_moving_variance"
assignvariableop_24_fc1_kernel 
assignvariableop_25_fc1_bias"
assignvariableop_26_fc2_kernel 
assignvariableop_27_fc2_bias"
assignvariableop_28_fc3_kernel 
assignvariableop_29_fc3_bias%
!assignvariableop_30_adadelta_iter&
"assignvariableop_31_adadelta_decay.
*assignvariableop_32_adadelta_learning_rate$
 assignvariableop_33_adadelta_rho
assignvariableop_34_total
assignvariableop_35_count
assignvariableop_36_total_1
assignvariableop_37_count_18
4assignvariableop_38_adadelta_conv1_kernel_accum_grad6
2assignvariableop_39_adadelta_conv1_bias_accum_grad:
6assignvariableop_40_adadelta_bn_conv1_gamma_accum_grad9
5assignvariableop_41_adadelta_bn_conv1_beta_accum_grad8
4assignvariableop_42_adadelta_conv2_kernel_accum_grad6
2assignvariableop_43_adadelta_conv2_bias_accum_grad:
6assignvariableop_44_adadelta_bn_conv2_gamma_accum_grad9
5assignvariableop_45_adadelta_bn_conv2_beta_accum_grad8
4assignvariableop_46_adadelta_conv3_kernel_accum_grad6
2assignvariableop_47_adadelta_conv3_bias_accum_grad:
6assignvariableop_48_adadelta_bn_conv3_gamma_accum_grad9
5assignvariableop_49_adadelta_bn_conv3_beta_accum_grad8
4assignvariableop_50_adadelta_conv4_kernel_accum_grad6
2assignvariableop_51_adadelta_conv4_bias_accum_grad:
6assignvariableop_52_adadelta_bn_conv4_gamma_accum_grad9
5assignvariableop_53_adadelta_bn_conv4_beta_accum_grad6
2assignvariableop_54_adadelta_fc1_kernel_accum_grad4
0assignvariableop_55_adadelta_fc1_bias_accum_grad6
2assignvariableop_56_adadelta_fc2_kernel_accum_grad4
0assignvariableop_57_adadelta_fc2_bias_accum_grad6
2assignvariableop_58_adadelta_fc3_kernel_accum_grad4
0assignvariableop_59_adadelta_fc3_bias_accum_grad7
3assignvariableop_60_adadelta_conv1_kernel_accum_var5
1assignvariableop_61_adadelta_conv1_bias_accum_var9
5assignvariableop_62_adadelta_bn_conv1_gamma_accum_var8
4assignvariableop_63_adadelta_bn_conv1_beta_accum_var7
3assignvariableop_64_adadelta_conv2_kernel_accum_var5
1assignvariableop_65_adadelta_conv2_bias_accum_var9
5assignvariableop_66_adadelta_bn_conv2_gamma_accum_var8
4assignvariableop_67_adadelta_bn_conv2_beta_accum_var7
3assignvariableop_68_adadelta_conv3_kernel_accum_var5
1assignvariableop_69_adadelta_conv3_bias_accum_var9
5assignvariableop_70_adadelta_bn_conv3_gamma_accum_var8
4assignvariableop_71_adadelta_bn_conv3_beta_accum_var7
3assignvariableop_72_adadelta_conv4_kernel_accum_var5
1assignvariableop_73_adadelta_conv4_bias_accum_var9
5assignvariableop_74_adadelta_bn_conv4_gamma_accum_var8
4assignvariableop_75_adadelta_bn_conv4_beta_accum_var5
1assignvariableop_76_adadelta_fc1_kernel_accum_var3
/assignvariableop_77_adadelta_fc1_bias_accum_var5
1assignvariableop_78_adadelta_fc2_kernel_accum_var3
/assignvariableop_79_adadelta_fc2_bias_accum_var5
1assignvariableop_80_adadelta_fc3_kernel_accum_var3
/assignvariableop_81_adadelta_fc3_bias_accum_var
identity_83??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_9?1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?0
value?0B?0SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_bn_conv1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_bn_conv1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp'assignvariableop_4_bn_conv1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_bn_conv1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_bn_conv2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_bn_conv2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp(assignvariableop_10_bn_conv2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_bn_conv2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_bn_conv3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_bn_conv3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_bn_conv3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_bn_conv3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_conv4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_bn_conv4_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_bn_conv4_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_bn_conv4_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_bn_conv4_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_fc1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_fc1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_fc2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_fc2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_fc3_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_fc3_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp!assignvariableop_30_adadelta_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_adadelta_decayIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adadelta_learning_rateIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp assignvariableop_33_adadelta_rhoIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_totalIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_countIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adadelta_conv1_kernel_accum_gradIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adadelta_conv1_bias_accum_gradIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adadelta_bn_conv1_gamma_accum_gradIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adadelta_bn_conv1_beta_accum_gradIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adadelta_conv2_kernel_accum_gradIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adadelta_conv2_bias_accum_gradIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adadelta_bn_conv2_gamma_accum_gradIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adadelta_bn_conv2_beta_accum_gradIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adadelta_conv3_kernel_accum_gradIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adadelta_conv3_bias_accum_gradIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adadelta_bn_conv3_gamma_accum_gradIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adadelta_bn_conv3_beta_accum_gradIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adadelta_conv4_kernel_accum_gradIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp2assignvariableop_51_adadelta_conv4_bias_accum_gradIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adadelta_bn_conv4_gamma_accum_gradIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adadelta_bn_conv4_beta_accum_gradIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adadelta_fc1_kernel_accum_gradIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp0assignvariableop_55_adadelta_fc1_bias_accum_gradIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adadelta_fc2_kernel_accum_gradIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp0assignvariableop_57_adadelta_fc2_bias_accum_gradIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp2assignvariableop_58_adadelta_fc3_kernel_accum_gradIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp0assignvariableop_59_adadelta_fc3_bias_accum_gradIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adadelta_conv1_kernel_accum_varIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp1assignvariableop_61_adadelta_conv1_bias_accum_varIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adadelta_bn_conv1_gamma_accum_varIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adadelta_bn_conv1_beta_accum_varIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp3assignvariableop_64_adadelta_conv2_kernel_accum_varIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp1assignvariableop_65_adadelta_conv2_bias_accum_varIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adadelta_bn_conv2_gamma_accum_varIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp4assignvariableop_67_adadelta_bn_conv2_beta_accum_varIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp3assignvariableop_68_adadelta_conv3_kernel_accum_varIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp1assignvariableop_69_adadelta_conv3_bias_accum_varIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adadelta_bn_conv3_gamma_accum_varIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp4assignvariableop_71_adadelta_bn_conv3_beta_accum_varIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp3assignvariableop_72_adadelta_conv4_kernel_accum_varIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp1assignvariableop_73_adadelta_conv4_bias_accum_varIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adadelta_bn_conv4_gamma_accum_varIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp4assignvariableop_75_adadelta_bn_conv4_beta_accum_varIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp1assignvariableop_76_adadelta_fc1_kernel_accum_varIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp/assignvariableop_77_adadelta_fc1_bias_accum_varIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp1assignvariableop_78_adadelta_fc2_kernel_accum_varIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp/assignvariableop_79_adadelta_fc2_bias_accum_varIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp1assignvariableop_80_adadelta_fc3_kernel_accum_varIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp/assignvariableop_81_adadelta_fc3_bias_accum_varIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82?
Identity_83IdentityIdentity_82:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_83"#
identity_83Identity_83:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
(__inference_bn_conv4_layer_call_fn_38584

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_364692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_36121

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
z
%__inference_conv2_layer_call_fn_38152

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_366072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv3_layer_call_fn_38400

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_368032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
>__inference_fc3_layer_call_and_return_conditional_losses_38747

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_36138

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_37048

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_activation_layer_call_fn_38133

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_365882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<<@:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_36722

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_38128

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????<<@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<<@:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv4_layer_call_fn_38648

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_369462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38540

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_36865

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38172

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
@__inference_conv1_layer_call_and_return_conditional_losses_37986

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
H
,__inference_activation_1_layer_call_fn_38290

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_367012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv1_layer_call_fn_38110

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_365292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????<<@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
??
?$
__inference__traced_save_39025
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop-
)savev2_bn_conv1_gamma_read_readvariableop,
(savev2_bn_conv1_beta_read_readvariableop3
/savev2_bn_conv1_moving_mean_read_readvariableop7
3savev2_bn_conv1_moving_variance_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop-
)savev2_bn_conv2_gamma_read_readvariableop,
(savev2_bn_conv2_beta_read_readvariableop3
/savev2_bn_conv2_moving_mean_read_readvariableop7
3savev2_bn_conv2_moving_variance_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop-
)savev2_bn_conv3_gamma_read_readvariableop,
(savev2_bn_conv3_beta_read_readvariableop3
/savev2_bn_conv3_moving_mean_read_readvariableop7
3savev2_bn_conv3_moving_variance_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop-
)savev2_bn_conv4_gamma_read_readvariableop,
(savev2_bn_conv4_beta_read_readvariableop3
/savev2_bn_conv4_moving_mean_read_readvariableop7
3savev2_bn_conv4_moving_variance_read_readvariableop)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop)
%savev2_fc2_kernel_read_readvariableop'
#savev2_fc2_bias_read_readvariableop)
%savev2_fc3_kernel_read_readvariableop'
#savev2_fc3_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_adadelta_conv1_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_conv1_bias_accum_grad_read_readvariableopA
=savev2_adadelta_bn_conv1_gamma_accum_grad_read_readvariableop@
<savev2_adadelta_bn_conv1_beta_accum_grad_read_readvariableop?
;savev2_adadelta_conv2_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_conv2_bias_accum_grad_read_readvariableopA
=savev2_adadelta_bn_conv2_gamma_accum_grad_read_readvariableop@
<savev2_adadelta_bn_conv2_beta_accum_grad_read_readvariableop?
;savev2_adadelta_conv3_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_conv3_bias_accum_grad_read_readvariableopA
=savev2_adadelta_bn_conv3_gamma_accum_grad_read_readvariableop@
<savev2_adadelta_bn_conv3_beta_accum_grad_read_readvariableop?
;savev2_adadelta_conv4_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_conv4_bias_accum_grad_read_readvariableopA
=savev2_adadelta_bn_conv4_gamma_accum_grad_read_readvariableop@
<savev2_adadelta_bn_conv4_beta_accum_grad_read_readvariableop=
9savev2_adadelta_fc1_kernel_accum_grad_read_readvariableop;
7savev2_adadelta_fc1_bias_accum_grad_read_readvariableop=
9savev2_adadelta_fc2_kernel_accum_grad_read_readvariableop;
7savev2_adadelta_fc2_bias_accum_grad_read_readvariableop=
9savev2_adadelta_fc3_kernel_accum_grad_read_readvariableop;
7savev2_adadelta_fc3_bias_accum_grad_read_readvariableop>
:savev2_adadelta_conv1_kernel_accum_var_read_readvariableop<
8savev2_adadelta_conv1_bias_accum_var_read_readvariableop@
<savev2_adadelta_bn_conv1_gamma_accum_var_read_readvariableop?
;savev2_adadelta_bn_conv1_beta_accum_var_read_readvariableop>
:savev2_adadelta_conv2_kernel_accum_var_read_readvariableop<
8savev2_adadelta_conv2_bias_accum_var_read_readvariableop@
<savev2_adadelta_bn_conv2_gamma_accum_var_read_readvariableop?
;savev2_adadelta_bn_conv2_beta_accum_var_read_readvariableop>
:savev2_adadelta_conv3_kernel_accum_var_read_readvariableop<
8savev2_adadelta_conv3_bias_accum_var_read_readvariableop@
<savev2_adadelta_bn_conv3_gamma_accum_var_read_readvariableop?
;savev2_adadelta_bn_conv3_beta_accum_var_read_readvariableop>
:savev2_adadelta_conv4_kernel_accum_var_read_readvariableop<
8savev2_adadelta_conv4_bias_accum_var_read_readvariableop@
<savev2_adadelta_bn_conv4_gamma_accum_var_read_readvariableop?
;savev2_adadelta_bn_conv4_beta_accum_var_read_readvariableop<
8savev2_adadelta_fc1_kernel_accum_var_read_readvariableop:
6savev2_adadelta_fc1_bias_accum_var_read_readvariableop<
8savev2_adadelta_fc2_kernel_accum_var_read_readvariableop:
6savev2_adadelta_fc2_bias_accum_var_read_readvariableop<
8savev2_adadelta_fc3_kernel_accum_var_read_readvariableop:
6savev2_adadelta_fc3_bias_accum_var_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?1
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?0
value?0B?0SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop)savev2_bn_conv1_gamma_read_readvariableop(savev2_bn_conv1_beta_read_readvariableop/savev2_bn_conv1_moving_mean_read_readvariableop3savev2_bn_conv1_moving_variance_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop)savev2_bn_conv2_gamma_read_readvariableop(savev2_bn_conv2_beta_read_readvariableop/savev2_bn_conv2_moving_mean_read_readvariableop3savev2_bn_conv2_moving_variance_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop)savev2_bn_conv3_gamma_read_readvariableop(savev2_bn_conv3_beta_read_readvariableop/savev2_bn_conv3_moving_mean_read_readvariableop3savev2_bn_conv3_moving_variance_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop)savev2_bn_conv4_gamma_read_readvariableop(savev2_bn_conv4_beta_read_readvariableop/savev2_bn_conv4_moving_mean_read_readvariableop3savev2_bn_conv4_moving_variance_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop%savev2_fc2_kernel_read_readvariableop#savev2_fc2_bias_read_readvariableop%savev2_fc3_kernel_read_readvariableop#savev2_fc3_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_adadelta_conv1_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv1_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv1_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv1_beta_accum_grad_read_readvariableop;savev2_adadelta_conv2_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv2_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv2_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv2_beta_accum_grad_read_readvariableop;savev2_adadelta_conv3_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv3_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv3_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv3_beta_accum_grad_read_readvariableop;savev2_adadelta_conv4_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv4_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv4_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv4_beta_accum_grad_read_readvariableop9savev2_adadelta_fc1_kernel_accum_grad_read_readvariableop7savev2_adadelta_fc1_bias_accum_grad_read_readvariableop9savev2_adadelta_fc2_kernel_accum_grad_read_readvariableop7savev2_adadelta_fc2_bias_accum_grad_read_readvariableop9savev2_adadelta_fc3_kernel_accum_grad_read_readvariableop7savev2_adadelta_fc3_bias_accum_grad_read_readvariableop:savev2_adadelta_conv1_kernel_accum_var_read_readvariableop8savev2_adadelta_conv1_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv1_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv1_beta_accum_var_read_readvariableop:savev2_adadelta_conv2_kernel_accum_var_read_readvariableop8savev2_adadelta_conv2_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv2_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv2_beta_accum_var_read_readvariableop:savev2_adadelta_conv3_kernel_accum_var_read_readvariableop8savev2_adadelta_conv3_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv3_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv3_beta_accum_var_read_readvariableop:savev2_adadelta_conv4_kernel_accum_var_read_readvariableop8savev2_adadelta_conv4_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv4_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv4_beta_accum_var_read_readvariableop8savev2_adadelta_fc1_kernel_accum_var_read_readvariableop6savev2_adadelta_fc1_bias_accum_var_read_readvariableop8savev2_adadelta_fc2_kernel_accum_var_read_readvariableop6savev2_adadelta_fc2_bias_accum_var_read_readvariableop8savev2_adadelta_fc3_kernel_accum_var_read_readvariableop6savev2_adadelta_fc3_bias_accum_var_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:@:@:@@:@:@:@:@:@:@?:?:?:?:?:?:??:?:?:?:?:?:
?	?:?:
??:?:	?:: : : : : : : : :@:@:@:@:@@:@:@:@:@?:?:?:?:??:?:?:?:
?	?:?:
??:?:	?::@:@:@:@:@@:@:@:@:@?:?:?:?:??:?:?:?:
?	?:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
?	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :,'(
&
_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@:,+(
&
_output_shapes
:@@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@:-/)
'
_output_shapes
:@?:!0

_output_shapes	
:?:!1

_output_shapes	
:?:!2

_output_shapes	
:?:.3*
(
_output_shapes
:??:!4

_output_shapes	
:?:!5

_output_shapes	
:?:!6

_output_shapes	
:?:&7"
 
_output_shapes
:
?	?:!8

_output_shapes	
:?:&9"
 
_output_shapes
:
??:!:

_output_shapes	
:?:%;!

_output_shapes
:	?: <

_output_shapes
::,=(
&
_output_shapes
:@: >

_output_shapes
:@: ?

_output_shapes
:@: @

_output_shapes
:@:,A(
&
_output_shapes
:@@: B

_output_shapes
:@: C

_output_shapes
:@: D

_output_shapes
:@:-E)
'
_output_shapes
:@?:!F

_output_shapes	
:?:!G

_output_shapes	
:?:!H

_output_shapes	
:?:.I*
(
_output_shapes
:??:!J

_output_shapes	
:?:!K

_output_shapes	
:?:!L

_output_shapes	
:?:&M"
 
_output_shapes
:
?	?:!N

_output_shapes	
:?:&O"
 
_output_shapes
:
??:!P

_output_shapes	
:?:%Q!

_output_shapes
:	?: R

_output_shapes
::S

_output_shapes
: 
?
?
)__inference_CNN_Model_layer_call_fn_37911

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_372962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
z
%__inference_conv3_layer_call_fn_38336

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_367502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_36144

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_361382
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_36438

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
>__inference_fc2_layer_call_and_return_conditional_losses_38727

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_38312

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_367222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_38706

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv2_layer_call_fn_38216

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_366602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_CNN_Model_layer_call_fn_37359
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_372962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
(__inference_bn_conv1_layer_call_fn_38046

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_360902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
x
#__inference_fc1_layer_call_fn_38689

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_370202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38236

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_38664

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
@__inference_conv4_layer_call_and_return_conditional_losses_38511

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_38716

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_370532
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_36547

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????<<@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????<<@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38033

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
@__inference_conv4_layer_call_and_return_conditional_losses_36893

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv2_layer_call_fn_38267

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_362062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38097

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????<<@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????<<@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_38701

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_36028
input_12
.cnn_model_conv1_conv2d_readvariableop_resource3
/cnn_model_conv1_biasadd_readvariableop_resource.
*cnn_model_bn_conv1_readvariableop_resource0
,cnn_model_bn_conv1_readvariableop_1_resource?
;cnn_model_bn_conv1_fusedbatchnormv3_readvariableop_resourceA
=cnn_model_bn_conv1_fusedbatchnormv3_readvariableop_1_resource2
.cnn_model_conv2_conv2d_readvariableop_resource3
/cnn_model_conv2_biasadd_readvariableop_resource.
*cnn_model_bn_conv2_readvariableop_resource0
,cnn_model_bn_conv2_readvariableop_1_resource?
;cnn_model_bn_conv2_fusedbatchnormv3_readvariableop_resourceA
=cnn_model_bn_conv2_fusedbatchnormv3_readvariableop_1_resource2
.cnn_model_conv3_conv2d_readvariableop_resource3
/cnn_model_conv3_biasadd_readvariableop_resource.
*cnn_model_bn_conv3_readvariableop_resource0
,cnn_model_bn_conv3_readvariableop_1_resource?
;cnn_model_bn_conv3_fusedbatchnormv3_readvariableop_resourceA
=cnn_model_bn_conv3_fusedbatchnormv3_readvariableop_1_resource2
.cnn_model_conv4_conv2d_readvariableop_resource3
/cnn_model_conv4_biasadd_readvariableop_resource.
*cnn_model_bn_conv4_readvariableop_resource0
,cnn_model_bn_conv4_readvariableop_1_resource?
;cnn_model_bn_conv4_fusedbatchnormv3_readvariableop_resourceA
=cnn_model_bn_conv4_fusedbatchnormv3_readvariableop_1_resource0
,cnn_model_fc1_matmul_readvariableop_resource1
-cnn_model_fc1_biasadd_readvariableop_resource0
,cnn_model_fc2_matmul_readvariableop_resource1
-cnn_model_fc2_biasadd_readvariableop_resource0
,cnn_model_fc3_matmul_readvariableop_resource1
-cnn_model_fc3_biasadd_readvariableop_resource
identity??2CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv1/ReadVariableOp?#CNN_Model/bn_conv1/ReadVariableOp_1?2CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv2/ReadVariableOp?#CNN_Model/bn_conv2/ReadVariableOp_1?2CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv3/ReadVariableOp?#CNN_Model/bn_conv3/ReadVariableOp_1?2CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv4/ReadVariableOp?#CNN_Model/bn_conv4/ReadVariableOp_1?&CNN_Model/conv1/BiasAdd/ReadVariableOp?%CNN_Model/conv1/Conv2D/ReadVariableOp?&CNN_Model/conv2/BiasAdd/ReadVariableOp?%CNN_Model/conv2/Conv2D/ReadVariableOp?&CNN_Model/conv3/BiasAdd/ReadVariableOp?%CNN_Model/conv3/Conv2D/ReadVariableOp?&CNN_Model/conv4/BiasAdd/ReadVariableOp?%CNN_Model/conv4/Conv2D/ReadVariableOp?$CNN_Model/fc1/BiasAdd/ReadVariableOp?#CNN_Model/fc1/MatMul/ReadVariableOp?$CNN_Model/fc2/BiasAdd/ReadVariableOp?#CNN_Model/fc2/MatMul/ReadVariableOp?$CNN_Model/fc3/BiasAdd/ReadVariableOp?#CNN_Model/fc3/MatMul/ReadVariableOp?
%CNN_Model/conv1/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02'
%CNN_Model/conv1/Conv2D/ReadVariableOp?
CNN_Model/conv1/Conv2DConv2Dinput_1-CNN_Model/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
CNN_Model/conv1/Conv2D?
&CNN_Model/conv1/BiasAdd/ReadVariableOpReadVariableOp/cnn_model_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&CNN_Model/conv1/BiasAdd/ReadVariableOp?
CNN_Model/conv1/BiasAddBiasAddCNN_Model/conv1/Conv2D:output:0.CNN_Model/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2
CNN_Model/conv1/BiasAdd?
!CNN_Model/bn_conv1/ReadVariableOpReadVariableOp*cnn_model_bn_conv1_readvariableop_resource*
_output_shapes
:@*
dtype02#
!CNN_Model/bn_conv1/ReadVariableOp?
#CNN_Model/bn_conv1/ReadVariableOp_1ReadVariableOp,cnn_model_bn_conv1_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#CNN_Model/bn_conv1/ReadVariableOp_1?
2CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOpReadVariableOp;cnn_model_bn_conv1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp?
4CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cnn_model_bn_conv1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_1?
#CNN_Model/bn_conv1/FusedBatchNormV3FusedBatchNormV3 CNN_Model/conv1/BiasAdd:output:0)CNN_Model/bn_conv1/ReadVariableOp:value:0+CNN_Model/bn_conv1/ReadVariableOp_1:value:0:CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp:value:0<CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????<<@:@:@:@:@:*
epsilon%o?:*
is_training( 2%
#CNN_Model/bn_conv1/FusedBatchNormV3?
CNN_Model/activation/ReluRelu'CNN_Model/bn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????<<@2
CNN_Model/activation/Relu?
CNN_Model/max_pooling2d/MaxPoolMaxPool'CNN_Model/activation/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2!
CNN_Model/max_pooling2d/MaxPool?
%CNN_Model/conv2/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02'
%CNN_Model/conv2/Conv2D/ReadVariableOp?
CNN_Model/conv2/Conv2DConv2D(CNN_Model/max_pooling2d/MaxPool:output:0-CNN_Model/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
CNN_Model/conv2/Conv2D?
&CNN_Model/conv2/BiasAdd/ReadVariableOpReadVariableOp/cnn_model_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&CNN_Model/conv2/BiasAdd/ReadVariableOp?
CNN_Model/conv2/BiasAddBiasAddCNN_Model/conv2/Conv2D:output:0.CNN_Model/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
CNN_Model/conv2/BiasAdd?
!CNN_Model/bn_conv2/ReadVariableOpReadVariableOp*cnn_model_bn_conv2_readvariableop_resource*
_output_shapes
:@*
dtype02#
!CNN_Model/bn_conv2/ReadVariableOp?
#CNN_Model/bn_conv2/ReadVariableOp_1ReadVariableOp,cnn_model_bn_conv2_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#CNN_Model/bn_conv2/ReadVariableOp_1?
2CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOpReadVariableOp;cnn_model_bn_conv2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp?
4CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cnn_model_bn_conv2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_1?
#CNN_Model/bn_conv2/FusedBatchNormV3FusedBatchNormV3 CNN_Model/conv2/BiasAdd:output:0)CNN_Model/bn_conv2/ReadVariableOp:value:0+CNN_Model/bn_conv2/ReadVariableOp_1:value:0:CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp:value:0<CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2%
#CNN_Model/bn_conv2/FusedBatchNormV3?
CNN_Model/activation_1/ReluRelu'CNN_Model/bn_conv2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
CNN_Model/activation_1/Relu?
!CNN_Model/max_pooling2d_1/MaxPoolMaxPool)CNN_Model/activation_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2#
!CNN_Model/max_pooling2d_1/MaxPool?
CNN_Model/dropout/IdentityIdentity*CNN_Model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
CNN_Model/dropout/Identity?
%CNN_Model/conv3/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02'
%CNN_Model/conv3/Conv2D/ReadVariableOp?
CNN_Model/conv3/Conv2DConv2D#CNN_Model/dropout/Identity:output:0-CNN_Model/conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CNN_Model/conv3/Conv2D?
&CNN_Model/conv3/BiasAdd/ReadVariableOpReadVariableOp/cnn_model_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&CNN_Model/conv3/BiasAdd/ReadVariableOp?
CNN_Model/conv3/BiasAddBiasAddCNN_Model/conv3/Conv2D:output:0.CNN_Model/conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CNN_Model/conv3/BiasAdd?
!CNN_Model/bn_conv3/ReadVariableOpReadVariableOp*cnn_model_bn_conv3_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!CNN_Model/bn_conv3/ReadVariableOp?
#CNN_Model/bn_conv3/ReadVariableOp_1ReadVariableOp,cnn_model_bn_conv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#CNN_Model/bn_conv3/ReadVariableOp_1?
2CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOpReadVariableOp;cnn_model_bn_conv3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp?
4CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cnn_model_bn_conv3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_1?
#CNN_Model/bn_conv3/FusedBatchNormV3FusedBatchNormV3 CNN_Model/conv3/BiasAdd:output:0)CNN_Model/bn_conv3/ReadVariableOp:value:0+CNN_Model/bn_conv3/ReadVariableOp_1:value:0:CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp:value:0<CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2%
#CNN_Model/bn_conv3/FusedBatchNormV3?
CNN_Model/activation_2/ReluRelu'CNN_Model/bn_conv3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
CNN_Model/activation_2/Relu?
!CNN_Model/max_pooling2d_2/MaxPoolMaxPool)CNN_Model/activation_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2#
!CNN_Model/max_pooling2d_2/MaxPool?
CNN_Model/dropout_1/IdentityIdentity*CNN_Model/max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
CNN_Model/dropout_1/Identity?
%CNN_Model/conv4/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02'
%CNN_Model/conv4/Conv2D/ReadVariableOp?
CNN_Model/conv4/Conv2DConv2D%CNN_Model/dropout_1/Identity:output:0-CNN_Model/conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CNN_Model/conv4/Conv2D?
&CNN_Model/conv4/BiasAdd/ReadVariableOpReadVariableOp/cnn_model_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&CNN_Model/conv4/BiasAdd/ReadVariableOp?
CNN_Model/conv4/BiasAddBiasAddCNN_Model/conv4/Conv2D:output:0.CNN_Model/conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CNN_Model/conv4/BiasAdd?
!CNN_Model/bn_conv4/ReadVariableOpReadVariableOp*cnn_model_bn_conv4_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!CNN_Model/bn_conv4/ReadVariableOp?
#CNN_Model/bn_conv4/ReadVariableOp_1ReadVariableOp,cnn_model_bn_conv4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#CNN_Model/bn_conv4/ReadVariableOp_1?
2CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOpReadVariableOp;cnn_model_bn_conv4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp?
4CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cnn_model_bn_conv4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_1?
#CNN_Model/bn_conv4/FusedBatchNormV3FusedBatchNormV3 CNN_Model/conv4/BiasAdd:output:0)CNN_Model/bn_conv4/ReadVariableOp:value:0+CNN_Model/bn_conv4/ReadVariableOp_1:value:0:CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp:value:0<CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2%
#CNN_Model/bn_conv4/FusedBatchNormV3?
CNN_Model/activation_3/ReluRelu'CNN_Model/bn_conv4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
CNN_Model/activation_3/Relu?
CNN_Model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
CNN_Model/flatten/Const?
CNN_Model/flatten/ReshapeReshape)CNN_Model/activation_3/Relu:activations:0 CNN_Model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
CNN_Model/flatten/Reshape?
#CNN_Model/fc1/MatMul/ReadVariableOpReadVariableOp,cnn_model_fc1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02%
#CNN_Model/fc1/MatMul/ReadVariableOp?
CNN_Model/fc1/MatMulMatMul"CNN_Model/flatten/Reshape:output:0+CNN_Model/fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_Model/fc1/MatMul?
$CNN_Model/fc1/BiasAdd/ReadVariableOpReadVariableOp-cnn_model_fc1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$CNN_Model/fc1/BiasAdd/ReadVariableOp?
CNN_Model/fc1/BiasAddBiasAddCNN_Model/fc1/MatMul:product:0,CNN_Model/fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_Model/fc1/BiasAdd?
CNN_Model/fc1/ReluReluCNN_Model/fc1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
CNN_Model/fc1/Relu?
CNN_Model/dropout_2/IdentityIdentity CNN_Model/fc1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
CNN_Model/dropout_2/Identity?
#CNN_Model/fc2/MatMul/ReadVariableOpReadVariableOp,cnn_model_fc2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#CNN_Model/fc2/MatMul/ReadVariableOp?
CNN_Model/fc2/MatMulMatMul%CNN_Model/dropout_2/Identity:output:0+CNN_Model/fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_Model/fc2/MatMul?
$CNN_Model/fc2/BiasAdd/ReadVariableOpReadVariableOp-cnn_model_fc2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$CNN_Model/fc2/BiasAdd/ReadVariableOp?
CNN_Model/fc2/BiasAddBiasAddCNN_Model/fc2/MatMul:product:0,CNN_Model/fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CNN_Model/fc2/BiasAdd?
CNN_Model/fc2/ReluReluCNN_Model/fc2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
CNN_Model/fc2/Relu?
#CNN_Model/fc3/MatMul/ReadVariableOpReadVariableOp,cnn_model_fc3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#CNN_Model/fc3/MatMul/ReadVariableOp?
CNN_Model/fc3/MatMulMatMul CNN_Model/fc2/Relu:activations:0+CNN_Model/fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_Model/fc3/MatMul?
$CNN_Model/fc3/BiasAdd/ReadVariableOpReadVariableOp-cnn_model_fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$CNN_Model/fc3/BiasAdd/ReadVariableOp?
CNN_Model/fc3/BiasAddBiasAddCNN_Model/fc3/MatMul:product:0,CNN_Model/fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CNN_Model/fc3/BiasAdd?
CNN_Model/fc3/SoftmaxSoftmaxCNN_Model/fc3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
CNN_Model/fc3/Softmax?

IdentityIdentityCNN_Model/fc3/Softmax:softmax:03^CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv1/ReadVariableOp$^CNN_Model/bn_conv1/ReadVariableOp_13^CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv2/ReadVariableOp$^CNN_Model/bn_conv2/ReadVariableOp_13^CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv3/ReadVariableOp$^CNN_Model/bn_conv3/ReadVariableOp_13^CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv4/ReadVariableOp$^CNN_Model/bn_conv4/ReadVariableOp_1'^CNN_Model/conv1/BiasAdd/ReadVariableOp&^CNN_Model/conv1/Conv2D/ReadVariableOp'^CNN_Model/conv2/BiasAdd/ReadVariableOp&^CNN_Model/conv2/Conv2D/ReadVariableOp'^CNN_Model/conv3/BiasAdd/ReadVariableOp&^CNN_Model/conv3/Conv2D/ReadVariableOp'^CNN_Model/conv4/BiasAdd/ReadVariableOp&^CNN_Model/conv4/Conv2D/ReadVariableOp%^CNN_Model/fc1/BiasAdd/ReadVariableOp$^CNN_Model/fc1/MatMul/ReadVariableOp%^CNN_Model/fc2/BiasAdd/ReadVariableOp$^CNN_Model/fc2/MatMul/ReadVariableOp%^CNN_Model/fc3/BiasAdd/ReadVariableOp$^CNN_Model/fc3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::2h
2CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp2CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp2l
4CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_14CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_12F
!CNN_Model/bn_conv1/ReadVariableOp!CNN_Model/bn_conv1/ReadVariableOp2J
#CNN_Model/bn_conv1/ReadVariableOp_1#CNN_Model/bn_conv1/ReadVariableOp_12h
2CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp2CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp2l
4CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_14CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_12F
!CNN_Model/bn_conv2/ReadVariableOp!CNN_Model/bn_conv2/ReadVariableOp2J
#CNN_Model/bn_conv2/ReadVariableOp_1#CNN_Model/bn_conv2/ReadVariableOp_12h
2CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp2CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp2l
4CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_14CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_12F
!CNN_Model/bn_conv3/ReadVariableOp!CNN_Model/bn_conv3/ReadVariableOp2J
#CNN_Model/bn_conv3/ReadVariableOp_1#CNN_Model/bn_conv3/ReadVariableOp_12h
2CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp2CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp2l
4CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_14CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_12F
!CNN_Model/bn_conv4/ReadVariableOp!CNN_Model/bn_conv4/ReadVariableOp2J
#CNN_Model/bn_conv4/ReadVariableOp_1#CNN_Model/bn_conv4/ReadVariableOp_12P
&CNN_Model/conv1/BiasAdd/ReadVariableOp&CNN_Model/conv1/BiasAdd/ReadVariableOp2N
%CNN_Model/conv1/Conv2D/ReadVariableOp%CNN_Model/conv1/Conv2D/ReadVariableOp2P
&CNN_Model/conv2/BiasAdd/ReadVariableOp&CNN_Model/conv2/BiasAdd/ReadVariableOp2N
%CNN_Model/conv2/Conv2D/ReadVariableOp%CNN_Model/conv2/Conv2D/ReadVariableOp2P
&CNN_Model/conv3/BiasAdd/ReadVariableOp&CNN_Model/conv3/BiasAdd/ReadVariableOp2N
%CNN_Model/conv3/Conv2D/ReadVariableOp%CNN_Model/conv3/Conv2D/ReadVariableOp2P
&CNN_Model/conv4/BiasAdd/ReadVariableOp&CNN_Model/conv4/BiasAdd/ReadVariableOp2N
%CNN_Model/conv4/Conv2D/ReadVariableOp%CNN_Model/conv4/Conv2D/ReadVariableOp2L
$CNN_Model/fc1/BiasAdd/ReadVariableOp$CNN_Model/fc1/BiasAdd/ReadVariableOp2J
#CNN_Model/fc1/MatMul/ReadVariableOp#CNN_Model/fc1/MatMul/ReadVariableOp2L
$CNN_Model/fc2/BiasAdd/ReadVariableOp$CNN_Model/fc2/BiasAdd/ReadVariableOp2J
#CNN_Model/fc2/MatMul/ReadVariableOp#CNN_Model/fc2/MatMul/ReadVariableOp2L
$CNN_Model/fc3/BiasAdd/ReadVariableOp$CNN_Model/fc3/BiasAdd/ReadVariableOp2J
#CNN_Model/fc3/MatMul/ReadVariableOp#CNN_Model/fc3/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
H
,__inference_activation_3_layer_call_fn_38658

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_369872
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
@__inference_conv2_layer_call_and_return_conditional_losses_36607

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38254

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv3_layer_call_fn_38451

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_363222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_38491

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38374

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv3_layer_call_fn_38464

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_363532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?_
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37296

inputs
conv1_37213
conv1_37215
bn_conv1_37218
bn_conv1_37220
bn_conv1_37222
bn_conv1_37224
conv2_37229
conv2_37231
bn_conv2_37234
bn_conv2_37236
bn_conv2_37238
bn_conv2_37240
conv3_37246
conv3_37248
bn_conv3_37251
bn_conv3_37253
bn_conv3_37255
bn_conv3_37257
conv4_37263
conv4_37265
bn_conv4_37268
bn_conv4_37270
bn_conv4_37272
bn_conv4_37274
	fc1_37279
	fc1_37281
	fc2_37285
	fc2_37287
	fc3_37290
	fc3_37292
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_37213conv1_37215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_364942
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_37218bn_conv1_37220bn_conv1_37222bn_conv1_37224*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_365292"
 bn_conv1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_365882
activation/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_361382
max_pooling2d/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_37229conv2_37231*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_366072
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_37234bn_conv2_37236bn_conv2_37238bn_conv2_37240*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_366422"
 bn_conv2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_367012
activation_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_362542!
max_pooling2d_1/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_367222!
dropout/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv3_37246conv3_37248*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_367502
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_37251bn_conv3_37253bn_conv3_37255bn_conv3_37257*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_367852"
 bn_conv3/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_368442
activation_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_363702!
max_pooling2d_2/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_368652#
!dropout_1/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv4_37263conv4_37265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_368932
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_37268bn_conv4_37270bn_conv4_37272bn_conv4_37274*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_369282"
 bn_conv4/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_369872
activation_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_370012
flatten/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	fc1_37279	fc1_37281*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_370202
fc1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_370482#
!dropout_2/StatefulPartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0	fc2_37285	fc2_37287*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_370772
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_37290	fc3_37292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc3_layer_call_and_return_conditional_losses_371042
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_36946

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_36870

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_38317

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_367272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
@__inference_conv3_layer_call_and_return_conditional_losses_38327

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv4_layer_call_fn_38635

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_369282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_36928

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_36260

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_362542
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_36370

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_37583
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_360282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_36254

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_38307

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_36353

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
@__inference_conv1_layer_call_and_return_conditional_losses_36494

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv3_layer_call_fn_38387

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_367852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38420

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_38496

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_368652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_38711

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_370482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_36469

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_38653

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
@__inference_conv2_layer_call_and_return_conditional_losses_38143

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv1_layer_call_fn_38123

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_365472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????<<@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_36642

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_CNN_Model_layer_call_fn_37510
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_374472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38438

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_36701

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38079

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????<<@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????<<@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?_
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37121
input_1
conv1_36505
conv1_36507
bn_conv1_36574
bn_conv1_36576
bn_conv1_36578
bn_conv1_36580
conv2_36618
conv2_36620
bn_conv2_36687
bn_conv2_36689
bn_conv2_36691
bn_conv2_36693
conv3_36761
conv3_36763
bn_conv3_36830
bn_conv3_36832
bn_conv3_36834
bn_conv3_36836
conv4_36904
conv4_36906
bn_conv4_36973
bn_conv4_36975
bn_conv4_36977
bn_conv4_36979
	fc1_37031
	fc1_37033
	fc2_37088
	fc2_37090
	fc3_37115
	fc3_37117
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_36505conv1_36507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_364942
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_36574bn_conv1_36576bn_conv1_36578bn_conv1_36580*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_365292"
 bn_conv1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_365882
activation/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_361382
max_pooling2d/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_36618conv2_36620*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_366072
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_36687bn_conv2_36689bn_conv2_36691bn_conv2_36693*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_366422"
 bn_conv2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_367012
activation_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_362542!
max_pooling2d_1/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_367222!
dropout/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv3_36761conv3_36763*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_367502
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_36830bn_conv3_36832bn_conv3_36834bn_conv3_36836*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_367852"
 bn_conv3/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_368442
activation_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_363702!
max_pooling2d_2/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_368652#
!dropout_1/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv4_36904conv4_36906*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_368932
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_36973bn_conv4_36975bn_conv4_36977bn_conv4_36979*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_369282"
 bn_conv4/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_369872
activation_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_370012
flatten/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	fc1_37031	fc1_37033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_370202
fc1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_370482#
!dropout_2/StatefulPartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0	fc2_37088	fc2_37090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_370772
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_37115	fc3_37117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc3_layer_call_and_return_conditional_losses_371042
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_36785

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38356

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
x
#__inference_fc3_layer_call_fn_38756

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc3_layer_call_and_return_conditional_losses_371042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_38302

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_36529

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????<<@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????<<@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?
?
)__inference_CNN_Model_layer_call_fn_37976

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_374472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_36660

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
z
%__inference_conv4_layer_call_fn_38520

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_368932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv1_layer_call_fn_38059

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_361212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_36206

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
>__inference_fc1_layer_call_and_return_conditional_losses_37020

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
>__inference_fc1_layer_call_and_return_conditional_losses_38680

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_36803

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
݄
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37846

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource$
 bn_conv1_readvariableop_resource&
"bn_conv1_readvariableop_1_resource5
1bn_conv1_fusedbatchnormv3_readvariableop_resource7
3bn_conv1_fusedbatchnormv3_readvariableop_1_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource$
 bn_conv2_readvariableop_resource&
"bn_conv2_readvariableop_1_resource5
1bn_conv2_fusedbatchnormv3_readvariableop_resource7
3bn_conv2_fusedbatchnormv3_readvariableop_1_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource$
 bn_conv3_readvariableop_resource&
"bn_conv3_readvariableop_1_resource5
1bn_conv3_fusedbatchnormv3_readvariableop_resource7
3bn_conv3_fusedbatchnormv3_readvariableop_1_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource$
 bn_conv4_readvariableop_resource&
"bn_conv4_readvariableop_1_resource5
1bn_conv4_fusedbatchnormv3_readvariableop_resource7
3bn_conv4_fusedbatchnormv3_readvariableop_1_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identity??(bn_conv1/FusedBatchNormV3/ReadVariableOp?*bn_conv1/FusedBatchNormV3/ReadVariableOp_1?bn_conv1/ReadVariableOp?bn_conv1/ReadVariableOp_1?(bn_conv2/FusedBatchNormV3/ReadVariableOp?*bn_conv2/FusedBatchNormV3/ReadVariableOp_1?bn_conv2/ReadVariableOp?bn_conv2/ReadVariableOp_1?(bn_conv3/FusedBatchNormV3/ReadVariableOp?*bn_conv3/FusedBatchNormV3/ReadVariableOp_1?bn_conv3/ReadVariableOp?bn_conv3/ReadVariableOp_1?(bn_conv4/FusedBatchNormV3/ReadVariableOp?*bn_conv4/FusedBatchNormV3/ReadVariableOp_1?bn_conv4/ReadVariableOp?bn_conv4/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?fc1/BiasAdd/ReadVariableOp?fc1/MatMul/ReadVariableOp?fc2/BiasAdd/ReadVariableOp?fc2/MatMul/ReadVariableOp?fc3/BiasAdd/ReadVariableOp?fc3/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2
conv1/BiasAdd?
bn_conv1/ReadVariableOpReadVariableOp bn_conv1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_conv1/ReadVariableOp?
bn_conv1/ReadVariableOp_1ReadVariableOp"bn_conv1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_conv1/ReadVariableOp_1?
(bn_conv1/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(bn_conv1/FusedBatchNormV3/ReadVariableOp?
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1?
bn_conv1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn_conv1/ReadVariableOp:value:0!bn_conv1/ReadVariableOp_1:value:00bn_conv1/FusedBatchNormV3/ReadVariableOp:value:02bn_conv1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????<<@:@:@:@:@:*
epsilon%o?:*
is_training( 2
bn_conv1/FusedBatchNormV3?
activation/ReluRelubn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????<<@2
activation/Relu?
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2/BiasAdd?
bn_conv2/ReadVariableOpReadVariableOp bn_conv2_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_conv2/ReadVariableOp?
bn_conv2/ReadVariableOp_1ReadVariableOp"bn_conv2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_conv2/ReadVariableOp_1?
(bn_conv2/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(bn_conv2/FusedBatchNormV3/ReadVariableOp?
*bn_conv2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*bn_conv2/FusedBatchNormV3/ReadVariableOp_1?
bn_conv2/FusedBatchNormV3FusedBatchNormV3conv2/BiasAdd:output:0bn_conv2/ReadVariableOp:value:0!bn_conv2/ReadVariableOp_1:value:00bn_conv2/FusedBatchNormV3/ReadVariableOp:value:02bn_conv2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
bn_conv2/FusedBatchNormV3?
activation_1/ReluRelubn_conv2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
dropout/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout/Identity?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Ddropout/Identity:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3/BiasAdd?
bn_conv3/ReadVariableOpReadVariableOp bn_conv3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_conv3/ReadVariableOp?
bn_conv3/ReadVariableOp_1ReadVariableOp"bn_conv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_conv3/ReadVariableOp_1?
(bn_conv3/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(bn_conv3/FusedBatchNormV3/ReadVariableOp?
*bn_conv3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02,
*bn_conv3/FusedBatchNormV3/ReadVariableOp_1?
bn_conv3/FusedBatchNormV3FusedBatchNormV3conv3/BiasAdd:output:0bn_conv3/ReadVariableOp:value:0!bn_conv3/ReadVariableOp_1:value:00bn_conv3/FusedBatchNormV3/ReadVariableOp:value:02bn_conv3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
bn_conv3/FusedBatchNormV3?
activation_2/ReluRelubn_conv3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
dropout_1/IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_1/Identity?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4/Conv2D/ReadVariableOp?
conv4/Conv2DConv2Ddropout_1/Identity:output:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv4/Conv2D?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv4/BiasAdd/ReadVariableOp?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4/BiasAdd?
bn_conv4/ReadVariableOpReadVariableOp bn_conv4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_conv4/ReadVariableOp?
bn_conv4/ReadVariableOp_1ReadVariableOp"bn_conv4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_conv4/ReadVariableOp_1?
(bn_conv4/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(bn_conv4/FusedBatchNormV3/ReadVariableOp?
*bn_conv4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02,
*bn_conv4/FusedBatchNormV3/ReadVariableOp_1?
bn_conv4/FusedBatchNormV3FusedBatchNormV3conv4/BiasAdd:output:0bn_conv4/ReadVariableOp:value:0!bn_conv4/ReadVariableOp_1:value:00bn_conv4/FusedBatchNormV3/ReadVariableOp:value:02bn_conv4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
bn_conv4/FusedBatchNormV3?
activation_3/ReluRelubn_conv4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapeactivation_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
fc1/MatMul/ReadVariableOp?

fc1/MatMulMatMulflatten/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

fc1/MatMul?
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
fc1/BiasAdd/ReadVariableOp?
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
fc1/BiasAdde
fc1/ReluRelufc1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

fc1/Relu
dropout_2/IdentityIdentityfc1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_2/Identity?
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
fc2/MatMul/ReadVariableOp?

fc2/MatMulMatMuldropout_2/Identity:output:0!fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

fc2/MatMul?
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
fc2/BiasAdd/ReadVariableOp?
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
fc2/BiasAdde
fc2/ReluRelufc2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

fc2/Relu?
fc3/MatMul/ReadVariableOpReadVariableOp"fc3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
fc3/MatMul/ReadVariableOp?

fc3/MatMulMatMulfc2/Relu:activations:0!fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

fc3/MatMul?
fc3/BiasAdd/ReadVariableOpReadVariableOp#fc3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc3/BiasAdd/ReadVariableOp?
fc3/BiasAddBiasAddfc3/MatMul:product:0"fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fc3/BiasAddm
fc3/SoftmaxSoftmaxfc3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
fc3/Softmax?
IdentityIdentityfc3/Softmax:softmax:0)^bn_conv1/FusedBatchNormV3/ReadVariableOp+^bn_conv1/FusedBatchNormV3/ReadVariableOp_1^bn_conv1/ReadVariableOp^bn_conv1/ReadVariableOp_1)^bn_conv2/FusedBatchNormV3/ReadVariableOp+^bn_conv2/FusedBatchNormV3/ReadVariableOp_1^bn_conv2/ReadVariableOp^bn_conv2/ReadVariableOp_1)^bn_conv3/FusedBatchNormV3/ReadVariableOp+^bn_conv3/FusedBatchNormV3/ReadVariableOp_1^bn_conv3/ReadVariableOp^bn_conv3/ReadVariableOp_1)^bn_conv4/FusedBatchNormV3/ReadVariableOp+^bn_conv4/FusedBatchNormV3/ReadVariableOp_1^bn_conv4/ReadVariableOp^bn_conv4/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp^fc3/BiasAdd/ReadVariableOp^fc3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::2T
(bn_conv1/FusedBatchNormV3/ReadVariableOp(bn_conv1/FusedBatchNormV3/ReadVariableOp2X
*bn_conv1/FusedBatchNormV3/ReadVariableOp_1*bn_conv1/FusedBatchNormV3/ReadVariableOp_122
bn_conv1/ReadVariableOpbn_conv1/ReadVariableOp26
bn_conv1/ReadVariableOp_1bn_conv1/ReadVariableOp_12T
(bn_conv2/FusedBatchNormV3/ReadVariableOp(bn_conv2/FusedBatchNormV3/ReadVariableOp2X
*bn_conv2/FusedBatchNormV3/ReadVariableOp_1*bn_conv2/FusedBatchNormV3/ReadVariableOp_122
bn_conv2/ReadVariableOpbn_conv2/ReadVariableOp26
bn_conv2/ReadVariableOp_1bn_conv2/ReadVariableOp_12T
(bn_conv3/FusedBatchNormV3/ReadVariableOp(bn_conv3/FusedBatchNormV3/ReadVariableOp2X
*bn_conv3/FusedBatchNormV3/ReadVariableOp_1*bn_conv3/FusedBatchNormV3/ReadVariableOp_122
bn_conv3/ReadVariableOpbn_conv3/ReadVariableOp26
bn_conv3/ReadVariableOp_1bn_conv3/ReadVariableOp_12T
(bn_conv4/FusedBatchNormV3/ReadVariableOp(bn_conv4/FusedBatchNormV3/ReadVariableOp2X
*bn_conv4/FusedBatchNormV3/ReadVariableOp_1*bn_conv4/FusedBatchNormV3/ReadVariableOp_122
bn_conv4/ReadVariableOpbn_conv4/ReadVariableOp26
bn_conv4/ReadVariableOp_1bn_conv4/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp28
fc3/BiasAdd/ReadVariableOpfc3/BiasAdd/ReadVariableOp26
fc3/MatMul/ReadVariableOpfc3/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38604

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
>__inference_fc2_layer_call_and_return_conditional_losses_37077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?[
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37447

inputs
conv1_37364
conv1_37366
bn_conv1_37369
bn_conv1_37371
bn_conv1_37373
bn_conv1_37375
conv2_37380
conv2_37382
bn_conv2_37385
bn_conv2_37387
bn_conv2_37389
bn_conv2_37391
conv3_37397
conv3_37399
bn_conv3_37402
bn_conv3_37404
bn_conv3_37406
bn_conv3_37408
conv4_37414
conv4_37416
bn_conv4_37419
bn_conv4_37421
bn_conv4_37423
bn_conv4_37425
	fc1_37430
	fc1_37432
	fc2_37436
	fc2_37438
	fc3_37441
	fc3_37443
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_37364conv1_37366*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_364942
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_37369bn_conv1_37371bn_conv1_37373bn_conv1_37375*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv1_layer_call_and_return_conditional_losses_365472"
 bn_conv1/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_365882
activation/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_361382
max_pooling2d/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2_37380conv2_37382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_366072
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_37385bn_conv2_37387bn_conv2_37389bn_conv2_37391*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_366602"
 bn_conv2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_367012
activation_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_362542!
max_pooling2d_1/PartitionedCall?
dropout/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_367272
dropout/PartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv3_37397conv3_37399*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv3_layer_call_and_return_conditional_losses_367502
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_37402bn_conv3_37404bn_conv3_37406bn_conv3_37408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv3_layer_call_and_return_conditional_losses_368032"
 bn_conv3/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_368442
activation_2/PartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_363702!
max_pooling2d_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_368702
dropout_1/PartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv4_37414conv4_37416*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv4_layer_call_and_return_conditional_losses_368932
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_37419bn_conv4_37421bn_conv4_37423bn_conv4_37425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv4_layer_call_and_return_conditional_losses_369462"
 bn_conv4/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_369872
activation_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_370012
flatten/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	fc1_37430	fc1_37432*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc1_layer_call_and_return_conditional_losses_370202
fc1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_370532
dropout_2/PartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0	fc2_37436	fc2_37438*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_370772
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_37441	fc3_37443*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc3_layer_call_and_return_conditional_losses_371042
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
@__inference_conv3_layer_call_and_return_conditional_losses_36750

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_38669

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_370012
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_36987

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_38469

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
z
%__inference_conv1_layer_call_fn_37995

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_364942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_36322

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_2_layer_call_fn_38474

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_368442
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38622

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38190

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv2_layer_call_fn_38203

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv2_layer_call_and_return_conditional_losses_366422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_36727

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_36588

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????<<@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<<@:W S
/
_output_shapes
:?????????<<@
 
_user_specified_nameinputs
?	
?
>__inference_fc3_layer_call_and_return_conditional_losses_37104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_38285

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_37001

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
x
#__inference_fc2_layer_call_fn_38736

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_fc2_layer_call_and_return_conditional_losses_370772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_38501

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_368702
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@@7
fc30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer-20
layer_with_weights-9
layer-21
layer_with_weights-10
layer-22
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_networkǡ{"class_name": "Functional", "name": "CNN_Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bn_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv3", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bn_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv4", "inbound_nodes": [[["conv4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["bn_conv4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc2", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc3", "inbound_nodes": [[["fc2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bn_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv3", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bn_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv4", "inbound_nodes": [[["conv4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["bn_conv4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc2", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc3", "inbound_nodes": [[["fc2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc3", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.05000000074505806, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?	
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 60, 64]}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
1trainable_variables
2	variables
3regularization_losses
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

5kernel
6bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 64]}}
?	
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 64]}}
?
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?	

Pkernel
Qbias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}
?	
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[trainable_variables
\	variables
]regularization_losses
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 128]}}
?
_trainable_variables
`	variables
aregularization_losses
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?	

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 128]}}
?	
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 128]}}
?
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
~trainable_variables
	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fc3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
	?iter

?decay
?learning_rate
?rho
accum_grad?
accum_grad?%
accum_grad?&
accum_grad?5
accum_grad?6
accum_grad?<
accum_grad?=
accum_grad?P
accum_grad?Q
accum_grad?W
accum_grad?X
accum_grad?k
accum_grad?l
accum_grad?r
accum_grad?s
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad?	accum_var?	accum_var?%	accum_var?&	accum_var?5	accum_var?6	accum_var?<	accum_var?=	accum_var?P	accum_var?Q	accum_var?W	accum_var?X	accum_var?k	accum_var?l	accum_var?r	accum_var?s	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var?"
	optimizer
?
0
1
%2
&3
54
65
<6
=7
P8
Q9
W10
X11
k12
l13
r14
s15
?16
?17
?18
?19
?20
?21"
trackable_list_wrapper
?
0
1
%2
&3
'4
(5
56
67
<8
=9
>10
?11
P12
Q13
W14
X15
Y16
Z17
k18
l19
r20
s21
t22
u23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?layers
 ?layer_regularization_losses
?metrics
	variables
?non_trainable_variables
regularization_losses
?layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$@2conv1/kernel
:@2
conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
 trainable_variables
 ?layer_regularization_losses
?layers
?metrics
!	variables
?non_trainable_variables
"regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2bn_conv1/gamma
:@2bn_conv1/beta
$:"@ (2bn_conv1/moving_mean
(:&@ (2bn_conv1/moving_variance
.
%0
&1"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)trainable_variables
 ?layer_regularization_losses
?layers
?metrics
*	variables
?non_trainable_variables
+regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-trainable_variables
 ?layer_regularization_losses
?layers
?metrics
.	variables
?non_trainable_variables
/regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1trainable_variables
 ?layer_regularization_losses
?layers
?metrics
2	variables
?non_trainable_variables
3regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv2/kernel
:@2
conv2/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7trainable_variables
 ?layer_regularization_losses
?layers
?metrics
8	variables
?non_trainable_variables
9regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2bn_conv2/gamma
:@2bn_conv2/beta
$:"@ (2bn_conv2/moving_mean
(:&@ (2bn_conv2/moving_variance
.
<0
=1"
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@trainable_variables
 ?layer_regularization_losses
?layers
?metrics
A	variables
?non_trainable_variables
Bregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
E	variables
?non_trainable_variables
Fregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Htrainable_variables
 ?layer_regularization_losses
?layers
?metrics
I	variables
?non_trainable_variables
Jregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ltrainable_variables
 ?layer_regularization_losses
?layers
?metrics
M	variables
?non_trainable_variables
Nregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@?2conv3/kernel
:?2
conv3/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
S	variables
?non_trainable_variables
Tregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2bn_conv3/gamma
:?2bn_conv3/beta
%:#? (2bn_conv3/moving_mean
):'? (2bn_conv3/moving_variance
.
W0
X1"
trackable_list_wrapper
<
W0
X1
Y2
Z3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[trainable_variables
 ?layer_regularization_losses
?layers
?metrics
\	variables
?non_trainable_variables
]regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
`	variables
?non_trainable_variables
aregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ctrainable_variables
 ?layer_regularization_losses
?layers
?metrics
d	variables
?non_trainable_variables
eregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
gtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
h	variables
?non_trainable_variables
iregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv4/kernel
:?2
conv4/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
n	variables
?non_trainable_variables
oregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2bn_conv4/gamma
:?2bn_conv4/beta
%:#? (2bn_conv4/moving_mean
):'? (2bn_conv4/moving_variance
.
r0
s1"
trackable_list_wrapper
<
r0
s1
t2
u3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vtrainable_variables
 ?layer_regularization_losses
?layers
?metrics
w	variables
?non_trainable_variables
xregularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ztrainable_variables
 ?layer_regularization_losses
?layers
?metrics
{	variables
?non_trainable_variables
|regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
~trainable_variables
 ?layer_regularization_losses
?layers
?metrics
	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
?	?2
fc1/kernel
:?2fc1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2
fc2/kernel
:?2fc2/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2
fc3/kernel
:2fc3/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?non_trainable_variables
?regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
X
'0
(1
>2
?3
Y4
Z5
t6
u7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
8:6@2 Adadelta/conv1/kernel/accum_grad
*:(@2Adadelta/conv1/bias/accum_grad
.:,@2"Adadelta/bn_conv1/gamma/accum_grad
-:+@2!Adadelta/bn_conv1/beta/accum_grad
8:6@@2 Adadelta/conv2/kernel/accum_grad
*:(@2Adadelta/conv2/bias/accum_grad
.:,@2"Adadelta/bn_conv2/gamma/accum_grad
-:+@2!Adadelta/bn_conv2/beta/accum_grad
9:7@?2 Adadelta/conv3/kernel/accum_grad
+:)?2Adadelta/conv3/bias/accum_grad
/:-?2"Adadelta/bn_conv3/gamma/accum_grad
.:,?2!Adadelta/bn_conv3/beta/accum_grad
::8??2 Adadelta/conv4/kernel/accum_grad
+:)?2Adadelta/conv4/bias/accum_grad
/:-?2"Adadelta/bn_conv4/gamma/accum_grad
.:,?2!Adadelta/bn_conv4/beta/accum_grad
0:.
?	?2Adadelta/fc1/kernel/accum_grad
):'?2Adadelta/fc1/bias/accum_grad
0:.
??2Adadelta/fc2/kernel/accum_grad
):'?2Adadelta/fc2/bias/accum_grad
/:-	?2Adadelta/fc3/kernel/accum_grad
(:&2Adadelta/fc3/bias/accum_grad
7:5@2Adadelta/conv1/kernel/accum_var
):'@2Adadelta/conv1/bias/accum_var
-:+@2!Adadelta/bn_conv1/gamma/accum_var
,:*@2 Adadelta/bn_conv1/beta/accum_var
7:5@@2Adadelta/conv2/kernel/accum_var
):'@2Adadelta/conv2/bias/accum_var
-:+@2!Adadelta/bn_conv2/gamma/accum_var
,:*@2 Adadelta/bn_conv2/beta/accum_var
8:6@?2Adadelta/conv3/kernel/accum_var
*:(?2Adadelta/conv3/bias/accum_var
.:,?2!Adadelta/bn_conv3/gamma/accum_var
-:+?2 Adadelta/bn_conv3/beta/accum_var
9:7??2Adadelta/conv4/kernel/accum_var
*:(?2Adadelta/conv4/bias/accum_var
.:,?2!Adadelta/bn_conv4/gamma/accum_var
-:+?2 Adadelta/bn_conv4/beta/accum_var
/:-
?	?2Adadelta/fc1/kernel/accum_var
(:&?2Adadelta/fc1/bias/accum_var
/:-
??2Adadelta/fc2/kernel/accum_var
(:&?2Adadelta/fc2/bias/accum_var
.:,	?2Adadelta/fc3/kernel/accum_var
':%2Adadelta/fc3/bias/accum_var
?2?
 __inference__wrapped_model_36028?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@@
?2?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37207
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37729
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37121
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37846?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_CNN_Model_layer_call_fn_37976
)__inference_CNN_Model_layer_call_fn_37510
)__inference_CNN_Model_layer_call_fn_37911
)__inference_CNN_Model_layer_call_fn_37359?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_conv1_layer_call_and_return_conditional_losses_37986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv1_layer_call_fn_37995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38015
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38079
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38033
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38097?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_bn_conv1_layer_call_fn_38046
(__inference_bn_conv1_layer_call_fn_38110
(__inference_bn_conv1_layer_call_fn_38123
(__inference_bn_conv1_layer_call_fn_38059?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_activation_layer_call_and_return_conditional_losses_38128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_activation_layer_call_fn_38133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_36138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_max_pooling2d_layer_call_fn_36144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
@__inference_conv2_layer_call_and_return_conditional_losses_38143?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv2_layer_call_fn_38152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38172
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38254
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38236
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38190?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_bn_conv2_layer_call_fn_38280
(__inference_bn_conv2_layer_call_fn_38203
(__inference_bn_conv2_layer_call_fn_38216
(__inference_bn_conv2_layer_call_fn_38267?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_1_layer_call_and_return_conditional_losses_38285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_1_layer_call_fn_38290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_36254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_1_layer_call_fn_36260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_38302
B__inference_dropout_layer_call_and_return_conditional_losses_38307?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_38317
'__inference_dropout_layer_call_fn_38312?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_conv3_layer_call_and_return_conditional_losses_38327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv3_layer_call_fn_38336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38356
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38420
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38438
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38374?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_bn_conv3_layer_call_fn_38387
(__inference_bn_conv3_layer_call_fn_38400
(__inference_bn_conv3_layer_call_fn_38464
(__inference_bn_conv3_layer_call_fn_38451?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_2_layer_call_and_return_conditional_losses_38469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_2_layer_call_fn_38474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_36370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_2_layer_call_fn_36376?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_38491
D__inference_dropout_1_layer_call_and_return_conditional_losses_38486?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_38501
)__inference_dropout_1_layer_call_fn_38496?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_conv4_layer_call_and_return_conditional_losses_38511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv4_layer_call_fn_38520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38622
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38558
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38540
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38604?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_bn_conv4_layer_call_fn_38571
(__inference_bn_conv4_layer_call_fn_38648
(__inference_bn_conv4_layer_call_fn_38584
(__inference_bn_conv4_layer_call_fn_38635?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_3_layer_call_and_return_conditional_losses_38653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_3_layer_call_fn_38658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_38664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_38669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_fc1_layer_call_and_return_conditional_losses_38680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_fc1_layer_call_fn_38689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_38706
D__inference_dropout_2_layer_call_and_return_conditional_losses_38701?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_2_layer_call_fn_38711
)__inference_dropout_2_layer_call_fn_38716?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_fc2_layer_call_and_return_conditional_losses_38727?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_fc2_layer_call_fn_38736?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_fc3_layer_call_and_return_conditional_losses_38747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_fc3_layer_call_fn_38756?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_37583input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37121?$%&'(56<=>?PQWXYZklrstu??????@?=
6?3
)?&
input_1?????????@@
p

 
? "%?"
?
0?????????
? ?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37207?$%&'(56<=>?PQWXYZklrstu??????@?=
6?3
)?&
input_1?????????@@
p 

 
? "%?"
?
0?????????
? ?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37729?$%&'(56<=>?PQWXYZklrstu????????<
5?2
(?%
inputs?????????@@
p

 
? "%?"
?
0?????????
? ?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_37846?$%&'(56<=>?PQWXYZklrstu????????<
5?2
(?%
inputs?????????@@
p 

 
? "%?"
?
0?????????
? ?
)__inference_CNN_Model_layer_call_fn_37359?$%&'(56<=>?PQWXYZklrstu??????@?=
6?3
)?&
input_1?????????@@
p

 
? "???????????
)__inference_CNN_Model_layer_call_fn_37510?$%&'(56<=>?PQWXYZklrstu??????@?=
6?3
)?&
input_1?????????@@
p 

 
? "???????????
)__inference_CNN_Model_layer_call_fn_37911?$%&'(56<=>?PQWXYZklrstu????????<
5?2
(?%
inputs?????????@@
p

 
? "???????????
)__inference_CNN_Model_layer_call_fn_37976?$%&'(56<=>?PQWXYZklrstu????????<
5?2
(?%
inputs?????????@@
p 

 
? "???????????
 __inference__wrapped_model_36028?$%&'(56<=>?PQWXYZklrstu??????8?5
.?+
)?&
input_1?????????@@
? ")?&
$
fc3?
fc3??????????
G__inference_activation_1_layer_call_and_return_conditional_losses_38285h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
,__inference_activation_1_layer_call_fn_38290[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
G__inference_activation_2_layer_call_and_return_conditional_losses_38469j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_activation_2_layer_call_fn_38474]8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_activation_3_layer_call_and_return_conditional_losses_38653j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_activation_3_layer_call_fn_38658]8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_activation_layer_call_and_return_conditional_losses_38128h7?4
-?*
(?%
inputs?????????<<@
? "-?*
#? 
0?????????<<@
? ?
*__inference_activation_layer_call_fn_38133[7?4
-?*
(?%
inputs?????????<<@
? " ??????????<<@?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38015?%&'(M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38033?%&'(M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38079r%&'(;?8
1?.
(?%
inputs?????????<<@
p
? "-?*
#? 
0?????????<<@
? ?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_38097r%&'(;?8
1?.
(?%
inputs?????????<<@
p 
? "-?*
#? 
0?????????<<@
? ?
(__inference_bn_conv1_layer_call_fn_38046?%&'(M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
(__inference_bn_conv1_layer_call_fn_38059?%&'(M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
(__inference_bn_conv1_layer_call_fn_38110e%&'(;?8
1?.
(?%
inputs?????????<<@
p
? " ??????????<<@?
(__inference_bn_conv1_layer_call_fn_38123e%&'(;?8
1?.
(?%
inputs?????????<<@
p 
? " ??????????<<@?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38172r<=>?;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38190r<=>?;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38236?<=>?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_38254?<=>?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
(__inference_bn_conv2_layer_call_fn_38203e<=>?;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
(__inference_bn_conv2_layer_call_fn_38216e<=>?;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
(__inference_bn_conv2_layer_call_fn_38267?<=>?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
(__inference_bn_conv2_layer_call_fn_38280?<=>?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38356tWXYZ<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38374tWXYZ<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38420?WXYZN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_38438?WXYZN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_bn_conv3_layer_call_fn_38387gWXYZ<?9
2?/
)?&
inputs??????????
p
? "!????????????
(__inference_bn_conv3_layer_call_fn_38400gWXYZ<?9
2?/
)?&
inputs??????????
p 
? "!????????????
(__inference_bn_conv3_layer_call_fn_38451?WXYZN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
(__inference_bn_conv3_layer_call_fn_38464?WXYZN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38540?rstuN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38558?rstuN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38604trstu<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_38622trstu<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
(__inference_bn_conv4_layer_call_fn_38571?rstuN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
(__inference_bn_conv4_layer_call_fn_38584?rstuN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
(__inference_bn_conv4_layer_call_fn_38635grstu<?9
2?/
)?&
inputs??????????
p
? "!????????????
(__inference_bn_conv4_layer_call_fn_38648grstu<?9
2?/
)?&
inputs??????????
p 
? "!????????????
@__inference_conv1_layer_call_and_return_conditional_losses_37986l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????<<@
? ?
%__inference_conv1_layer_call_fn_37995_7?4
-?*
(?%
inputs?????????@@
? " ??????????<<@?
@__inference_conv2_layer_call_and_return_conditional_losses_38143l567?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
%__inference_conv2_layer_call_fn_38152_567?4
-?*
(?%
inputs?????????@
? " ??????????@?
@__inference_conv3_layer_call_and_return_conditional_losses_38327mPQ7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
%__inference_conv3_layer_call_fn_38336`PQ7?4
-?*
(?%
inputs?????????@
? "!????????????
@__inference_conv4_layer_call_and_return_conditional_losses_38511nkl8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
%__inference_conv4_layer_call_fn_38520akl8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_38486n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_38491n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
)__inference_dropout_1_layer_call_fn_38496a<?9
2?/
)?&
inputs??????????
p
? "!????????????
)__inference_dropout_1_layer_call_fn_38501a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
D__inference_dropout_2_layer_call_and_return_conditional_losses_38701^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_38706^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ~
)__inference_dropout_2_layer_call_fn_38711Q4?1
*?'
!?
inputs??????????
p
? "???????????~
)__inference_dropout_2_layer_call_fn_38716Q4?1
*?'
!?
inputs??????????
p 
? "????????????
B__inference_dropout_layer_call_and_return_conditional_losses_38302l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_38307l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
'__inference_dropout_layer_call_fn_38312_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
'__inference_dropout_layer_call_fn_38317_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
>__inference_fc1_layer_call_and_return_conditional_losses_38680`??0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? z
#__inference_fc1_layer_call_fn_38689S??0?-
&?#
!?
inputs??????????	
? "????????????
>__inference_fc2_layer_call_and_return_conditional_losses_38727`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
#__inference_fc2_layer_call_fn_38736S??0?-
&?#
!?
inputs??????????
? "????????????
>__inference_fc3_layer_call_and_return_conditional_losses_38747_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
#__inference_fc3_layer_call_fn_38756R??0?-
&?#
!?
inputs??????????
? "???????????
B__inference_flatten_layer_call_and_return_conditional_losses_38664b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????	
? ?
'__inference_flatten_layer_call_fn_38669U8?5
.?+
)?&
inputs??????????
? "???????????	?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_36254?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_36260?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_36370?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_36376?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_36138?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_36144?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
#__inference_signature_wrapper_37583?$%&'(56<=>?PQWXYZklrstu??????C?@
? 
9?6
4
input_1)?&
input_1?????????@@")?&
$
fc3?
fc3?????????
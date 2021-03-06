??"
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
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
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
~
conv5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv5/kernel
w
 conv5/kernel/Read/ReadVariableOpReadVariableOpconv5/kernel*(
_output_shapes
:??*
dtype0
m

conv5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv5/bias
f
conv5/bias/Read/ReadVariableOpReadVariableOp
conv5/bias*
_output_shapes	
:?*
dtype0
u
bn_conv5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebn_conv5/gamma
n
"bn_conv5/gamma/Read/ReadVariableOpReadVariableOpbn_conv5/gamma*
_output_shapes	
:?*
dtype0
s
bn_conv5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebn_conv5/beta
l
!bn_conv5/beta/Read/ReadVariableOpReadVariableOpbn_conv5/beta*
_output_shapes	
:?*
dtype0
?
bn_conv5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_conv5/moving_mean
z
(bn_conv5/moving_mean/Read/ReadVariableOpReadVariableOpbn_conv5/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_conv5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebn_conv5/moving_variance
?
,bn_conv5/moving_variance/Read/ReadVariableOpReadVariableOpbn_conv5/moving_variance*
_output_shapes	
:?*
dtype0
r

fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name
fc1/kernel
k
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel* 
_output_shapes
:
??*
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
 Adadelta/conv5/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*1
shared_name" Adadelta/conv5/kernel/accum_grad
?
4Adadelta/conv5/kernel/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/conv5/kernel/accum_grad*(
_output_shapes
:??*
dtype0
?
Adadelta/conv5/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adadelta/conv5/bias/accum_grad
?
2Adadelta/conv5/bias/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/conv5/bias/accum_grad*
_output_shapes	
:?*
dtype0
?
"Adadelta/bn_conv5/gamma/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adadelta/bn_conv5/gamma/accum_grad
?
6Adadelta/bn_conv5/gamma/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/bn_conv5/gamma/accum_grad*
_output_shapes	
:?*
dtype0
?
!Adadelta/bn_conv5/beta/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adadelta/bn_conv5/beta/accum_grad
?
5Adadelta/bn_conv5/beta/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv5/beta/accum_grad*
_output_shapes	
:?*
dtype0
?
Adadelta/fc1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adadelta/fc1/kernel/accum_grad
?
2Adadelta/fc1/kernel/accum_grad/Read/ReadVariableOpReadVariableOpAdadelta/fc1/kernel/accum_grad* 
_output_shapes
:
??*
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
Adadelta/conv5/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*0
shared_name!Adadelta/conv5/kernel/accum_var
?
3Adadelta/conv5/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv5/kernel/accum_var*(
_output_shapes
:??*
dtype0
?
Adadelta/conv5/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameAdadelta/conv5/bias/accum_var
?
1Adadelta/conv5/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/conv5/bias/accum_var*
_output_shapes	
:?*
dtype0
?
!Adadelta/bn_conv5/gamma/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adadelta/bn_conv5/gamma/accum_var
?
5Adadelta/bn_conv5/gamma/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/bn_conv5/gamma/accum_var*
_output_shapes	
:?*
dtype0
?
 Adadelta/bn_conv5/beta/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adadelta/bn_conv5/beta/accum_var
?
4Adadelta/bn_conv5/beta/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/bn_conv5/beta/accum_var*
_output_shapes	
:?*
dtype0
?
Adadelta/fc1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*.
shared_nameAdadelta/fc1/kernel/accum_var
?
1Adadelta/fc1/kernel/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/fc1/kernel/accum_var* 
_output_shapes
:
??*
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
dtype0*̷
value??B?? B??
?
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
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer-25
layer_with_weights-11
layer-26
layer_with_weights-12
layer-27
	optimizer
trainable_variables
regularization_losses
 	variables
!	keras_api
"
signatures
 
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
R
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
atrainable_variables
bregularization_losses
c	keras_api
R
d	variables
etrainable_variables
fregularization_losses
g	keras_api
R
h	variables
itrainable_variables
jregularization_losses
k	keras_api
R
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
h

pkernel
qbias
r	variables
strainable_variables
tregularization_losses
u	keras_api
?
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{	variables
|trainable_variables
}regularization_losses
~	keras_api
U
	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?iter

?decay
?learning_rate
?rho#
accum_grad?$
accum_grad?*
accum_grad?+
accum_grad?:
accum_grad?;
accum_grad?A
accum_grad?B
accum_grad?U
accum_grad?V
accum_grad?\
accum_grad?]
accum_grad?p
accum_grad?q
accum_grad?w
accum_grad?x
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad?#	accum_var?$	accum_var?*	accum_var?+	accum_var?:	accum_var?;	accum_var?A	accum_var?B	accum_var?U	accum_var?V	accum_var?\	accum_var?]	accum_var?p	accum_var?q	accum_var?w	accum_var?x	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var?
?
#0
$1
*2
+3
:4
;5
A6
B7
U8
V9
\10
]11
p12
q13
w14
x15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
 
?
#0
$1
*2
+3
,4
-5
:6
;7
A8
B9
C10
D11
U12
V13
\14
]15
^16
_17
p18
q19
w20
x21
y22
z23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?
?layers
?metrics
?layer_metrics
 ?layer_regularization_losses
trainable_variables
regularization_losses
?non_trainable_variables
 	variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
?layers
?metrics
%	variables
 ?layer_regularization_losses
&trainable_variables
'regularization_losses
?non_trainable_variables
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

*0
+1
,2
-3

*0
+1
 
?
?layers
?metrics
.	variables
 ?layer_regularization_losses
/trainable_variables
0regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
2	variables
 ?layer_regularization_losses
3trainable_variables
4regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
6	variables
 ?layer_regularization_losses
7trainable_variables
8regularization_losses
?non_trainable_variables
?layer_metrics
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
?
?layers
?metrics
<	variables
 ?layer_regularization_losses
=trainable_variables
>regularization_losses
?non_trainable_variables
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

A0
B1
C2
D3

A0
B1
 
?
?layers
?metrics
E	variables
 ?layer_regularization_losses
Ftrainable_variables
Gregularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
I	variables
 ?layer_regularization_losses
Jtrainable_variables
Kregularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
M	variables
 ?layer_regularization_losses
Ntrainable_variables
Oregularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
Q	variables
 ?layer_regularization_losses
Rtrainable_variables
Sregularization_losses
?non_trainable_variables
?layer_metrics
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
?
?layers
?metrics
W	variables
 ?layer_regularization_losses
Xtrainable_variables
Yregularization_losses
?non_trainable_variables
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

\0
]1
^2
_3

\0
]1
 
?
?layers
?metrics
`	variables
 ?layer_regularization_losses
atrainable_variables
bregularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
d	variables
 ?layer_regularization_losses
etrainable_variables
fregularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
h	variables
 ?layer_regularization_losses
itrainable_variables
jregularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
l	variables
 ?layer_regularization_losses
mtrainable_variables
nregularization_losses
?non_trainable_variables
?layer_metrics
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1

p0
q1
 
?
?layers
?metrics
r	variables
 ?layer_regularization_losses
strainable_variables
tregularization_losses
?non_trainable_variables
?layer_metrics
 
YW
VARIABLE_VALUEbn_conv4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbn_conv4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbn_conv4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn_conv4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

w0
x1
y2
z3

w0
x1
 
?
?layers
?metrics
{	variables
 ?layer_regularization_losses
|trainable_variables
}regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
XV
VARIABLE_VALUEconv5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
 
YW
VARIABLE_VALUEbn_conv5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbn_conv5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbn_conv5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEbn_conv5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
WU
VARIABLE_VALUE
fc1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
WU
VARIABLE_VALUE
fc2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
WU
VARIABLE_VALUE
fc3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEfc3/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
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
23
24
25
26
27

?0
?1
 
 
H
,0
-1
C2
D3
^4
_5
y6
z7
?8
?9
 
 
 
 
 
 
 
 

,0
-1
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
C0
D1
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
^0
_1
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
y0
z1
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

?0
?1
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
VARIABLE_VALUE Adadelta/conv5/kernel/accum_grad[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv5/bias/accum_gradYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/bn_conv5/gamma/accum_gradZlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv5/beta/accum_gradYlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/kernel/accum_grad\layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/bias/accum_gradZlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/kernel/accum_grad\layer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/bias/accum_gradZlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/kernel/accum_grad\layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/bias/accum_gradZlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdadelta/conv5/kernel/accum_varZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/conv5/bias/accum_varXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/bn_conv5/gamma/accum_varYlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/bn_conv5/beta/accum_varXlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/kernel/accum_var[layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc1/bias/accum_varYlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/kernel/accum_var[layer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc2/bias/accum_varYlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/kernel/accum_var[layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdadelta/fc3/bias/accum_varYlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_3Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv1/kernel
conv1/biasbn_conv1/gammabn_conv1/betabn_conv1/moving_meanbn_conv1/moving_varianceconv2/kernel
conv2/biasbn_conv2/gammabn_conv2/betabn_conv2/moving_meanbn_conv2/moving_varianceconv3/kernel
conv3/biasbn_conv3/gammabn_conv3/betabn_conv3/moving_meanbn_conv3/moving_varianceconv4/kernel
conv4/biasbn_conv4/gammabn_conv4/betabn_conv4/moving_meanbn_conv4/moving_varianceconv5/kernel
conv5/biasbn_conv5/gammabn_conv5/betabn_conv5/moving_meanbn_conv5/moving_variance
fc1/kernelfc1/bias
fc2/kernelfc2/bias
fc3/kernelfc3/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_60578
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp"bn_conv1/gamma/Read/ReadVariableOp!bn_conv1/beta/Read/ReadVariableOp(bn_conv1/moving_mean/Read/ReadVariableOp,bn_conv1/moving_variance/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp"bn_conv2/gamma/Read/ReadVariableOp!bn_conv2/beta/Read/ReadVariableOp(bn_conv2/moving_mean/Read/ReadVariableOp,bn_conv2/moving_variance/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp"bn_conv3/gamma/Read/ReadVariableOp!bn_conv3/beta/Read/ReadVariableOp(bn_conv3/moving_mean/Read/ReadVariableOp,bn_conv3/moving_variance/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp"bn_conv4/gamma/Read/ReadVariableOp!bn_conv4/beta/Read/ReadVariableOp(bn_conv4/moving_mean/Read/ReadVariableOp,bn_conv4/moving_variance/Read/ReadVariableOp conv5/kernel/Read/ReadVariableOpconv5/bias/Read/ReadVariableOp"bn_conv5/gamma/Read/ReadVariableOp!bn_conv5/beta/Read/ReadVariableOp(bn_conv5/moving_mean/Read/ReadVariableOp,bn_conv5/moving_variance/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOpfc2/kernel/Read/ReadVariableOpfc2/bias/Read/ReadVariableOpfc3/kernel/Read/ReadVariableOpfc3/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Adadelta/conv1/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv1/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv1/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv1/beta/accum_grad/Read/ReadVariableOp4Adadelta/conv2/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv2/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv2/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv2/beta/accum_grad/Read/ReadVariableOp4Adadelta/conv3/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv3/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv3/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv3/beta/accum_grad/Read/ReadVariableOp4Adadelta/conv4/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv4/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv4/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv4/beta/accum_grad/Read/ReadVariableOp4Adadelta/conv5/kernel/accum_grad/Read/ReadVariableOp2Adadelta/conv5/bias/accum_grad/Read/ReadVariableOp6Adadelta/bn_conv5/gamma/accum_grad/Read/ReadVariableOp5Adadelta/bn_conv5/beta/accum_grad/Read/ReadVariableOp2Adadelta/fc1/kernel/accum_grad/Read/ReadVariableOp0Adadelta/fc1/bias/accum_grad/Read/ReadVariableOp2Adadelta/fc2/kernel/accum_grad/Read/ReadVariableOp0Adadelta/fc2/bias/accum_grad/Read/ReadVariableOp2Adadelta/fc3/kernel/accum_grad/Read/ReadVariableOp0Adadelta/fc3/bias/accum_grad/Read/ReadVariableOp3Adadelta/conv1/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv1/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv1/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv1/beta/accum_var/Read/ReadVariableOp3Adadelta/conv2/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv2/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv2/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv2/beta/accum_var/Read/ReadVariableOp3Adadelta/conv3/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv3/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv3/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv3/beta/accum_var/Read/ReadVariableOp3Adadelta/conv4/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv4/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv4/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv4/beta/accum_var/Read/ReadVariableOp3Adadelta/conv5/kernel/accum_var/Read/ReadVariableOp1Adadelta/conv5/bias/accum_var/Read/ReadVariableOp5Adadelta/bn_conv5/gamma/accum_var/Read/ReadVariableOp4Adadelta/bn_conv5/beta/accum_var/Read/ReadVariableOp1Adadelta/fc1/kernel/accum_var/Read/ReadVariableOp/Adadelta/fc1/bias/accum_var/Read/ReadVariableOp1Adadelta/fc2/kernel/accum_var/Read/ReadVariableOp/Adadelta/fc2/bias/accum_var/Read/ReadVariableOp1Adadelta/fc3/kernel/accum_var/Read/ReadVariableOp/Adadelta/fc3/bias/accum_var/Read/ReadVariableOpConst*m
Tinf
d2b	*
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
__inference__traced_save_62325
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasbn_conv1/gammabn_conv1/betabn_conv1/moving_meanbn_conv1/moving_varianceconv2/kernel
conv2/biasbn_conv2/gammabn_conv2/betabn_conv2/moving_meanbn_conv2/moving_varianceconv3/kernel
conv3/biasbn_conv3/gammabn_conv3/betabn_conv3/moving_meanbn_conv3/moving_varianceconv4/kernel
conv4/biasbn_conv4/gammabn_conv4/betabn_conv4/moving_meanbn_conv4/moving_varianceconv5/kernel
conv5/biasbn_conv5/gammabn_conv5/betabn_conv5/moving_meanbn_conv5/moving_variance
fc1/kernelfc1/bias
fc2/kernelfc2/bias
fc3/kernelfc3/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_1 Adadelta/conv1/kernel/accum_gradAdadelta/conv1/bias/accum_grad"Adadelta/bn_conv1/gamma/accum_grad!Adadelta/bn_conv1/beta/accum_grad Adadelta/conv2/kernel/accum_gradAdadelta/conv2/bias/accum_grad"Adadelta/bn_conv2/gamma/accum_grad!Adadelta/bn_conv2/beta/accum_grad Adadelta/conv3/kernel/accum_gradAdadelta/conv3/bias/accum_grad"Adadelta/bn_conv3/gamma/accum_grad!Adadelta/bn_conv3/beta/accum_grad Adadelta/conv4/kernel/accum_gradAdadelta/conv4/bias/accum_grad"Adadelta/bn_conv4/gamma/accum_grad!Adadelta/bn_conv4/beta/accum_grad Adadelta/conv5/kernel/accum_gradAdadelta/conv5/bias/accum_grad"Adadelta/bn_conv5/gamma/accum_grad!Adadelta/bn_conv5/beta/accum_gradAdadelta/fc1/kernel/accum_gradAdadelta/fc1/bias/accum_gradAdadelta/fc2/kernel/accum_gradAdadelta/fc2/bias/accum_gradAdadelta/fc3/kernel/accum_gradAdadelta/fc3/bias/accum_gradAdadelta/conv1/kernel/accum_varAdadelta/conv1/bias/accum_var!Adadelta/bn_conv1/gamma/accum_var Adadelta/bn_conv1/beta/accum_varAdadelta/conv2/kernel/accum_varAdadelta/conv2/bias/accum_var!Adadelta/bn_conv2/gamma/accum_var Adadelta/bn_conv2/beta/accum_varAdadelta/conv3/kernel/accum_varAdadelta/conv3/bias/accum_var!Adadelta/bn_conv3/gamma/accum_var Adadelta/bn_conv3/beta/accum_varAdadelta/conv4/kernel/accum_varAdadelta/conv4/bias/accum_var!Adadelta/bn_conv4/gamma/accum_var Adadelta/bn_conv4/beta/accum_varAdadelta/conv5/kernel/accum_varAdadelta/conv5/bias/accum_var!Adadelta/bn_conv5/gamma/accum_var Adadelta/bn_conv5/beta/accum_varAdadelta/fc1/kernel/accum_varAdadelta/fc1/bias/accum_varAdadelta/fc2/kernel/accum_varAdadelta/fc2/bias/accum_varAdadelta/fc3/kernel/accum_varAdadelta/fc3/bias/accum_var*l
Tine
c2a*
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
!__inference__traced_restore_62623??
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_59961

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
?
d
H__inference_activation_12_layer_call_and_return_conditional_losses_59609

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
?
d
H__inference_activation_13_layer_call_and_return_conditional_losses_59752

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
?
z
%__inference_conv5_layer_call_fn_61778

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
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_598012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
>__inference_fc2_layer_call_and_return_conditional_losses_61985

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
?
?
(__inference_bn_conv4_layer_call_fn_61645

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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_596932
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
?	
?
>__inference_fc3_layer_call_and_return_conditional_losses_60012

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
?
z
%__inference_conv1_layer_call_fn_61069

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
@__inference_conv1_layer_call_and_return_conditional_losses_592592
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
?
?
(__inference_bn_conv5_layer_call_fn_61906

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
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_598542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv3_layer_call_fn_61525

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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_595502
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
C__inference_bn_conv5_layer_call_and_return_conditional_losses_59203

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_10_layer_call_and_return_conditional_losses_59353

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
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_58971

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
?
?
(__inference_bn_conv5_layer_call_fn_61829

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
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_592032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_13_layer_call_and_return_conditional_losses_61727

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
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_59487

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
?	
?
@__inference_conv4_layer_call_and_return_conditional_losses_59658

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
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61632

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
?o
?	
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60132
input_3
conv1_60032
conv1_60034
bn_conv1_60037
bn_conv1_60039
bn_conv1_60041
bn_conv1_60043
conv2_60048
conv2_60050
bn_conv2_60053
bn_conv2_60055
bn_conv2_60057
bn_conv2_60059
conv3_60065
conv3_60067
bn_conv3_60070
bn_conv3_60072
bn_conv3_60074
bn_conv3_60076
conv4_60082
conv4_60084
bn_conv4_60087
bn_conv4_60089
bn_conv4_60091
bn_conv4_60093
conv5_60099
conv5_60101
bn_conv5_60104
bn_conv5_60106
bn_conv5_60108
bn_conv5_60110
	fc1_60115
	fc1_60117
	fc2_60121
	fc2_60123
	fc3_60126
	fc3_60128
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall? bn_conv5/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv5/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_3conv1_60032conv1_60034*
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
@__inference_conv1_layer_call_and_return_conditional_losses_592592
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_60037bn_conv1_60039bn_conv1_60041bn_conv1_60043*
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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_593122"
 bn_conv1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_593532
activation_10/PartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_587872!
max_pooling2d_8/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2_60048conv2_60050*
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
@__inference_conv2_layer_call_and_return_conditional_losses_593722
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_60053bn_conv2_60055bn_conv2_60057bn_conv2_60059*
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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_594252"
 bn_conv2/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_594662
activation_11/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_589032!
max_pooling2d_9/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_594922
dropout_8/PartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv3_60065conv3_60067*
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
@__inference_conv3_layer_call_and_return_conditional_losses_595152
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_60070bn_conv3_60072bn_conv3_60074bn_conv3_60076*
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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_595682"
 bn_conv3/StatefulPartitionedCall?
activation_12/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_596092
activation_12/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_590192"
 max_pooling2d_10/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*
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
D__inference_dropout_9_layer_call_and_return_conditional_losses_596352
dropout_9/PartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv4_60082conv4_60084*
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
@__inference_conv4_layer_call_and_return_conditional_losses_596582
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_60087bn_conv4_60089bn_conv4_60091bn_conv4_60093*
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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_597112"
 bn_conv4/StatefulPartitionedCall?
activation_13/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_597522
activation_13/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_591352"
 max_pooling2d_11/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_597782
dropout_10/PartitionedCall?
conv5/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0conv5_60099conv5_60101*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_598012
conv5/StatefulPartitionedCall?
 bn_conv5/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0bn_conv5_60104bn_conv5_60106bn_conv5_60108bn_conv5_60110*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_598542"
 bn_conv5/StatefulPartitionedCall?
activation_14/PartitionedCallPartitionedCall)bn_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_598952
activation_14/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_599092
flatten_2/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0	fc1_60115	fc1_60117*
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
>__inference_fc1_layer_call_and_return_conditional_losses_599282
fc1/StatefulPartitionedCall?
dropout_11/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_599612
dropout_11/PartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0	fc2_60121	fc2_60123*
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
>__inference_fc2_layer_call_and_return_conditional_losses_599852
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_60126	fc3_60128*
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
>__inference_fc3_layer_call_and_return_conditional_losses_600122
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall!^bn_conv5/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2D
 bn_conv5/StatefulPartitionedCall bn_conv5/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_3
?
z
%__inference_conv2_layer_call_fn_61226

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
@__inference_conv2_layer_call_and_return_conditional_losses_593722
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
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_61749

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?u
?

D__inference_CNN_Model_layer_call_and_return_conditional_losses_60029
input_3
conv1_59270
conv1_59272
bn_conv1_59339
bn_conv1_59341
bn_conv1_59343
bn_conv1_59345
conv2_59383
conv2_59385
bn_conv2_59452
bn_conv2_59454
bn_conv2_59456
bn_conv2_59458
conv3_59526
conv3_59528
bn_conv3_59595
bn_conv3_59597
bn_conv3_59599
bn_conv3_59601
conv4_59669
conv4_59671
bn_conv4_59738
bn_conv4_59740
bn_conv4_59742
bn_conv4_59744
conv5_59812
conv5_59814
bn_conv5_59881
bn_conv5_59883
bn_conv5_59885
bn_conv5_59887
	fc1_59939
	fc1_59941
	fc2_59996
	fc2_59998
	fc3_60023
	fc3_60025
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall? bn_conv5/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv5/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinput_3conv1_59270conv1_59272*
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
@__inference_conv1_layer_call_and_return_conditional_losses_592592
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_59339bn_conv1_59341bn_conv1_59343bn_conv1_59345*
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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_592942"
 bn_conv1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_593532
activation_10/PartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_587872!
max_pooling2d_8/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2_59383conv2_59385*
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
@__inference_conv2_layer_call_and_return_conditional_losses_593722
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_59452bn_conv2_59454bn_conv2_59456bn_conv2_59458*
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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_594072"
 bn_conv2/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_594662
activation_11/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_589032!
max_pooling2d_9/PartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_594872#
!dropout_8/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv3_59526conv3_59528*
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
@__inference_conv3_layer_call_and_return_conditional_losses_595152
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_59595bn_conv3_59597bn_conv3_59599bn_conv3_59601*
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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_595502"
 bn_conv3/StatefulPartitionedCall?
activation_12/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_596092
activation_12/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_590192"
 max_pooling2d_10/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
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
D__inference_dropout_9_layer_call_and_return_conditional_losses_596302#
!dropout_9/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv4_59669conv4_59671*
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
@__inference_conv4_layer_call_and_return_conditional_losses_596582
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_59738bn_conv4_59740bn_conv4_59742bn_conv4_59744*
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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_596932"
 bn_conv4/StatefulPartitionedCall?
activation_13/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_597522
activation_13/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_591352"
 max_pooling2d_11/PartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_597732$
"dropout_10/StatefulPartitionedCall?
conv5/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0conv5_59812conv5_59814*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_598012
conv5/StatefulPartitionedCall?
 bn_conv5/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0bn_conv5_59881bn_conv5_59883bn_conv5_59885bn_conv5_59887*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_598362"
 bn_conv5/StatefulPartitionedCall?
activation_14/PartitionedCallPartitionedCall)bn_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_598952
activation_14/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_599092
flatten_2/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0	fc1_59939	fc1_59941*
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
>__inference_fc1_layer_call_and_return_conditional_losses_599282
fc1/StatefulPartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
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
GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_599562$
"dropout_11/StatefulPartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0	fc2_59996	fc2_59998*
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
>__inference_fc2_layer_call_and_return_conditional_losses_599852
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_60023	fc3_60025*
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
>__inference_fc3_layer_call_and_return_conditional_losses_600122
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall!^bn_conv5/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2D
 bn_conv5/StatefulPartitionedCall bn_conv5/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_3
?
d
H__inference_activation_12_layer_call_and_return_conditional_losses_61543

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
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_58739

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
??
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60896

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
3bn_conv4_fusedbatchnormv3_readvariableop_1_resource(
$conv5_conv2d_readvariableop_resource)
%conv5_biasadd_readvariableop_resource$
 bn_conv5_readvariableop_resource&
"bn_conv5_readvariableop_1_resource5
1bn_conv5_fusedbatchnormv3_readvariableop_resource7
3bn_conv5_fusedbatchnormv3_readvariableop_1_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identity??(bn_conv1/FusedBatchNormV3/ReadVariableOp?*bn_conv1/FusedBatchNormV3/ReadVariableOp_1?bn_conv1/ReadVariableOp?bn_conv1/ReadVariableOp_1?(bn_conv2/FusedBatchNormV3/ReadVariableOp?*bn_conv2/FusedBatchNormV3/ReadVariableOp_1?bn_conv2/ReadVariableOp?bn_conv2/ReadVariableOp_1?(bn_conv3/FusedBatchNormV3/ReadVariableOp?*bn_conv3/FusedBatchNormV3/ReadVariableOp_1?bn_conv3/ReadVariableOp?bn_conv3/ReadVariableOp_1?(bn_conv4/FusedBatchNormV3/ReadVariableOp?*bn_conv4/FusedBatchNormV3/ReadVariableOp_1?bn_conv4/ReadVariableOp?bn_conv4/ReadVariableOp_1?(bn_conv5/FusedBatchNormV3/ReadVariableOp?*bn_conv5/FusedBatchNormV3/ReadVariableOp_1?bn_conv5/ReadVariableOp?bn_conv5/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?conv5/BiasAdd/ReadVariableOp?conv5/Conv2D/ReadVariableOp?fc1/BiasAdd/ReadVariableOp?fc1/MatMul/ReadVariableOp?fc2/BiasAdd/ReadVariableOp?fc2/MatMul/ReadVariableOp?fc3/BiasAdd/ReadVariableOp?fc3/MatMul/ReadVariableOp?
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
activation_10/ReluRelubn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????<<@2
activation_10/Relu?
max_pooling2d_8/MaxPoolMaxPool activation_10/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2D max_pooling2d_8/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
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
activation_11/ReluRelubn_conv2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_11/Relu?
max_pooling2d_9/MaxPoolMaxPool activation_11/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool?
dropout_8/IdentityIdentity max_pooling2d_9/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_8/Identity?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Ddropout_8/Identity:output:0#conv3/Conv2D/ReadVariableOp:value:0*
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
activation_12/ReluRelubn_conv3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_12/Relu?
max_pooling2d_10/MaxPoolMaxPool activation_12/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool?
dropout_9/IdentityIdentity!max_pooling2d_10/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_9/Identity?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4/Conv2D/ReadVariableOp?
conv4/Conv2DConv2Ddropout_9/Identity:output:0#conv4/Conv2D/ReadVariableOp:value:0*
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
activation_13/ReluRelubn_conv4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_13/Relu?
max_pooling2d_11/MaxPoolMaxPool activation_13/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool?
dropout_10/IdentityIdentity!max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_10/Identity?
conv5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5/Conv2D/ReadVariableOp?
conv5/Conv2DConv2Ddropout_10/Identity:output:0#conv5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv5/Conv2D?
conv5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv5/BiasAdd/ReadVariableOp?
conv5/BiasAddBiasAddconv5/Conv2D:output:0$conv5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5/BiasAdd?
bn_conv5/ReadVariableOpReadVariableOp bn_conv5_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_conv5/ReadVariableOp?
bn_conv5/ReadVariableOp_1ReadVariableOp"bn_conv5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_conv5/ReadVariableOp_1?
(bn_conv5/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(bn_conv5/FusedBatchNormV3/ReadVariableOp?
*bn_conv5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02,
*bn_conv5/FusedBatchNormV3/ReadVariableOp_1?
bn_conv5/FusedBatchNormV3FusedBatchNormV3conv5/BiasAdd:output:0bn_conv5/ReadVariableOp:value:0!bn_conv5/ReadVariableOp_1:value:00bn_conv5/FusedBatchNormV3/ReadVariableOp:value:02bn_conv5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
bn_conv5/FusedBatchNormV3?
activation_14/ReluRelubn_conv5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_14/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshape activation_14/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
fc1/MatMul/ReadVariableOp?

fc1/MatMulMatMulflatten_2/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
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

fc1/Relu?
dropout_11/IdentityIdentityfc1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_11/Identity?
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
fc2/MatMul/ReadVariableOp?

fc2/MatMulMatMuldropout_11/Identity:output:0!fc2/MatMul/ReadVariableOp:value:0*
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

IdentityIdentityfc3/Softmax:softmax:0)^bn_conv1/FusedBatchNormV3/ReadVariableOp+^bn_conv1/FusedBatchNormV3/ReadVariableOp_1^bn_conv1/ReadVariableOp^bn_conv1/ReadVariableOp_1)^bn_conv2/FusedBatchNormV3/ReadVariableOp+^bn_conv2/FusedBatchNormV3/ReadVariableOp_1^bn_conv2/ReadVariableOp^bn_conv2/ReadVariableOp_1)^bn_conv3/FusedBatchNormV3/ReadVariableOp+^bn_conv3/FusedBatchNormV3/ReadVariableOp_1^bn_conv3/ReadVariableOp^bn_conv3/ReadVariableOp_1)^bn_conv4/FusedBatchNormV3/ReadVariableOp+^bn_conv4/FusedBatchNormV3/ReadVariableOp_1^bn_conv4/ReadVariableOp^bn_conv4/ReadVariableOp_1)^bn_conv5/FusedBatchNormV3/ReadVariableOp+^bn_conv5/FusedBatchNormV3/ReadVariableOp_1^bn_conv5/ReadVariableOp^bn_conv5/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^conv5/BiasAdd/ReadVariableOp^conv5/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp^fc3/BiasAdd/ReadVariableOp^fc3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::2T
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
bn_conv4/ReadVariableOp_1bn_conv4/ReadVariableOp_12T
(bn_conv5/FusedBatchNormV3/ReadVariableOp(bn_conv5/FusedBatchNormV3/ReadVariableOp2X
*bn_conv5/FusedBatchNormV3/ReadVariableOp_1*bn_conv5/FusedBatchNormV3/ReadVariableOp_122
bn_conv5/ReadVariableOpbn_conv5/ReadVariableOp26
bn_conv5/ReadVariableOp_1bn_conv5/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2<
conv5/BiasAdd/ReadVariableOpconv5/BiasAdd/ReadVariableOp2:
conv5/Conv2D/ReadVariableOpconv5/Conv2D/ReadVariableOp28
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
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61328

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
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61430

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
)__inference_dropout_8_layer_call_fn_61386

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
GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_594872
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
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_61922

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_CNN_Model_layer_call_fn_60313
input_3
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

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_602382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_3
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61089

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
>__inference_fc3_layer_call_and_return_conditional_losses_62005

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
?
?
(__inference_bn_conv2_layer_call_fn_61341

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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_594072
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
?
?
(__inference_bn_conv4_layer_call_fn_61722

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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_591182
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
?	
?
@__inference_conv5_layer_call_and_return_conditional_losses_59801

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_activation_12_layer_call_fn_61548

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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_596092
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
?
?
#__inference_signature_wrapper_60578
input_3
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

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_586772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_3
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_59492

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
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61246

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
?
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_59019

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
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_59956

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
?
c
*__inference_dropout_10_layer_call_fn_61754

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_597732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61880

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61798

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
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
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_61959

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
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61614

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
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_61560

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
??
?5
!__inference__traced_restore_62623
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
,assignvariableop_23_bn_conv4_moving_variance$
 assignvariableop_24_conv5_kernel"
assignvariableop_25_conv5_bias&
"assignvariableop_26_bn_conv5_gamma%
!assignvariableop_27_bn_conv5_beta,
(assignvariableop_28_bn_conv5_moving_mean0
,assignvariableop_29_bn_conv5_moving_variance"
assignvariableop_30_fc1_kernel 
assignvariableop_31_fc1_bias"
assignvariableop_32_fc2_kernel 
assignvariableop_33_fc2_bias"
assignvariableop_34_fc3_kernel 
assignvariableop_35_fc3_bias%
!assignvariableop_36_adadelta_iter&
"assignvariableop_37_adadelta_decay.
*assignvariableop_38_adadelta_learning_rate$
 assignvariableop_39_adadelta_rho
assignvariableop_40_total
assignvariableop_41_count
assignvariableop_42_total_1
assignvariableop_43_count_18
4assignvariableop_44_adadelta_conv1_kernel_accum_grad6
2assignvariableop_45_adadelta_conv1_bias_accum_grad:
6assignvariableop_46_adadelta_bn_conv1_gamma_accum_grad9
5assignvariableop_47_adadelta_bn_conv1_beta_accum_grad8
4assignvariableop_48_adadelta_conv2_kernel_accum_grad6
2assignvariableop_49_adadelta_conv2_bias_accum_grad:
6assignvariableop_50_adadelta_bn_conv2_gamma_accum_grad9
5assignvariableop_51_adadelta_bn_conv2_beta_accum_grad8
4assignvariableop_52_adadelta_conv3_kernel_accum_grad6
2assignvariableop_53_adadelta_conv3_bias_accum_grad:
6assignvariableop_54_adadelta_bn_conv3_gamma_accum_grad9
5assignvariableop_55_adadelta_bn_conv3_beta_accum_grad8
4assignvariableop_56_adadelta_conv4_kernel_accum_grad6
2assignvariableop_57_adadelta_conv4_bias_accum_grad:
6assignvariableop_58_adadelta_bn_conv4_gamma_accum_grad9
5assignvariableop_59_adadelta_bn_conv4_beta_accum_grad8
4assignvariableop_60_adadelta_conv5_kernel_accum_grad6
2assignvariableop_61_adadelta_conv5_bias_accum_grad:
6assignvariableop_62_adadelta_bn_conv5_gamma_accum_grad9
5assignvariableop_63_adadelta_bn_conv5_beta_accum_grad6
2assignvariableop_64_adadelta_fc1_kernel_accum_grad4
0assignvariableop_65_adadelta_fc1_bias_accum_grad6
2assignvariableop_66_adadelta_fc2_kernel_accum_grad4
0assignvariableop_67_adadelta_fc2_bias_accum_grad6
2assignvariableop_68_adadelta_fc3_kernel_accum_grad4
0assignvariableop_69_adadelta_fc3_bias_accum_grad7
3assignvariableop_70_adadelta_conv1_kernel_accum_var5
1assignvariableop_71_adadelta_conv1_bias_accum_var9
5assignvariableop_72_adadelta_bn_conv1_gamma_accum_var8
4assignvariableop_73_adadelta_bn_conv1_beta_accum_var7
3assignvariableop_74_adadelta_conv2_kernel_accum_var5
1assignvariableop_75_adadelta_conv2_bias_accum_var9
5assignvariableop_76_adadelta_bn_conv2_gamma_accum_var8
4assignvariableop_77_adadelta_bn_conv2_beta_accum_var7
3assignvariableop_78_adadelta_conv3_kernel_accum_var5
1assignvariableop_79_adadelta_conv3_bias_accum_var9
5assignvariableop_80_adadelta_bn_conv3_gamma_accum_var8
4assignvariableop_81_adadelta_bn_conv3_beta_accum_var7
3assignvariableop_82_adadelta_conv4_kernel_accum_var5
1assignvariableop_83_adadelta_conv4_bias_accum_var9
5assignvariableop_84_adadelta_bn_conv4_gamma_accum_var8
4assignvariableop_85_adadelta_bn_conv4_beta_accum_var7
3assignvariableop_86_adadelta_conv5_kernel_accum_var5
1assignvariableop_87_adadelta_conv5_bias_accum_var9
5assignvariableop_88_adadelta_bn_conv5_gamma_accum_var8
4assignvariableop_89_adadelta_bn_conv5_beta_accum_var5
1assignvariableop_90_adadelta_fc1_kernel_accum_var3
/assignvariableop_91_adadelta_fc1_bias_accum_var5
1assignvariableop_92_adadelta_fc2_kernel_accum_var3
/assignvariableop_93_adadelta_fc2_bias_accum_var5
1assignvariableop_94_adadelta_fc3_kernel_accum_var3
/assignvariableop_95_adadelta_fc3_bias_accum_var
identity_97??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?8
value?8B?8aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?
value?B?aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*o
dtypese
c2a	2
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
AssignVariableOp_24AssignVariableOp assignvariableop_24_conv5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_conv5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_bn_conv5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_bn_conv5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_bn_conv5_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_bn_conv5_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_fc1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_fc1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_fc2_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_fc2_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_fc3_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_fc3_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp!assignvariableop_36_adadelta_iterIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_adadelta_decayIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adadelta_learning_rateIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp assignvariableop_39_adadelta_rhoIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_total_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_count_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adadelta_conv1_kernel_accum_gradIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp2assignvariableop_45_adadelta_conv1_bias_accum_gradIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adadelta_bn_conv1_gamma_accum_gradIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adadelta_bn_conv1_beta_accum_gradIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp4assignvariableop_48_adadelta_conv2_kernel_accum_gradIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp2assignvariableop_49_adadelta_conv2_bias_accum_gradIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adadelta_bn_conv2_gamma_accum_gradIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adadelta_bn_conv2_beta_accum_gradIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adadelta_conv3_kernel_accum_gradIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adadelta_conv3_bias_accum_gradIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adadelta_bn_conv3_gamma_accum_gradIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adadelta_bn_conv3_beta_accum_gradIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp4assignvariableop_56_adadelta_conv4_kernel_accum_gradIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adadelta_conv4_bias_accum_gradIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adadelta_bn_conv4_gamma_accum_gradIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adadelta_bn_conv4_beta_accum_gradIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp4assignvariableop_60_adadelta_conv5_kernel_accum_gradIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adadelta_conv5_bias_accum_gradIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adadelta_bn_conv5_gamma_accum_gradIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp5assignvariableop_63_adadelta_bn_conv5_beta_accum_gradIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adadelta_fc1_kernel_accum_gradIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adadelta_fc1_bias_accum_gradIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp2assignvariableop_66_adadelta_fc2_kernel_accum_gradIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp0assignvariableop_67_adadelta_fc2_bias_accum_gradIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp2assignvariableop_68_adadelta_fc3_kernel_accum_gradIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp0assignvariableop_69_adadelta_fc3_bias_accum_gradIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp3assignvariableop_70_adadelta_conv1_kernel_accum_varIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp1assignvariableop_71_adadelta_conv1_bias_accum_varIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp5assignvariableop_72_adadelta_bn_conv1_gamma_accum_varIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp4assignvariableop_73_adadelta_bn_conv1_beta_accum_varIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp3assignvariableop_74_adadelta_conv2_kernel_accum_varIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp1assignvariableop_75_adadelta_conv2_bias_accum_varIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp5assignvariableop_76_adadelta_bn_conv2_gamma_accum_varIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp4assignvariableop_77_adadelta_bn_conv2_beta_accum_varIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp3assignvariableop_78_adadelta_conv3_kernel_accum_varIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp1assignvariableop_79_adadelta_conv3_bias_accum_varIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp5assignvariableop_80_adadelta_bn_conv3_gamma_accum_varIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp4assignvariableop_81_adadelta_bn_conv3_beta_accum_varIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp3assignvariableop_82_adadelta_conv4_kernel_accum_varIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp1assignvariableop_83_adadelta_conv4_bias_accum_varIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp5assignvariableop_84_adadelta_bn_conv4_gamma_accum_varIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp4assignvariableop_85_adadelta_bn_conv4_beta_accum_varIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp3assignvariableop_86_adadelta_conv5_kernel_accum_varIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp1assignvariableop_87_adadelta_conv5_bias_accum_varIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp5assignvariableop_88_adadelta_bn_conv5_gamma_accum_varIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp4assignvariableop_89_adadelta_bn_conv5_beta_accum_varIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp1assignvariableop_90_adadelta_fc1_kernel_accum_varIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp/assignvariableop_91_adadelta_fc1_bias_accum_varIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp1assignvariableop_92_adadelta_fc2_kernel_accum_varIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp/assignvariableop_93_adadelta_fc2_bias_accum_varIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp1assignvariableop_94_adadelta_fc3_kernel_accum_varIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp/assignvariableop_95_adadelta_fc3_bias_accum_varIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_959
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_96Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_96?
Identity_97IdentityIdentity_96:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95*
T0*
_output_shapes
: 2
Identity_97"#
identity_97Identity_97:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_95:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61494

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
?
d
H__inference_activation_11_layer_call_and_return_conditional_losses_59466

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
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_59711

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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_59568

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
?	
?
@__inference_conv2_layer_call_and_return_conditional_losses_61217

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
?
?
(__inference_bn_conv2_layer_call_fn_61277

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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_588552
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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_59002

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
@__inference_conv2_layer_call_and_return_conditional_losses_59372

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
(__inference_bn_conv4_layer_call_fn_61658

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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_597112
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
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_59909

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_58770

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
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61696

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
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61448

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
?
?
(__inference_bn_conv3_layer_call_fn_61538

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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_595682
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
?
z
%__inference_conv3_layer_call_fn_61410

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
@__inference_conv3_layer_call_and_return_conditional_losses_595152
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
?
?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_59854

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_activation_13_layer_call_fn_61732

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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_597522
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
?
?
(__inference_bn_conv5_layer_call_fn_61893

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
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_598362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_CNN_Model_layer_call_fn_60493
input_3
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

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_604182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_3
?	
?
@__inference_conv3_layer_call_and_return_conditional_losses_59515

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
?
L
0__inference_max_pooling2d_11_layer_call_fn_59141

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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_591352
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
?
x
#__inference_fc3_layer_call_fn_62014

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
>__inference_fc3_layer_call_and_return_conditional_losses_600122
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
?
>__inference_fc1_layer_call_and_return_conditional_losses_61938

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_59773

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_59234

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
x
#__inference_fc1_layer_call_fn_61947

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
>__inference_fc1_layer_call_and_return_conditional_losses_599282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_59135

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
?
I
-__inference_activation_14_layer_call_fn_61916

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_598952
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
@__inference_conv3_layer_call_and_return_conditional_losses_61401

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
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61678

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
E
)__inference_dropout_8_layer_call_fn_61391

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
GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_594922
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
?
?
(__inference_bn_conv2_layer_call_fn_61290

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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_588862
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
?
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_61964

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
(__inference_bn_conv1_layer_call_fn_61197

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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_593122
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
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_58855

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
?
?
)__inference_CNN_Model_layer_call_fn_60973

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

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_602382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
d
H__inference_activation_14_layer_call_and_return_conditional_losses_59895

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_59087

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
F
*__inference_dropout_11_layer_call_fn_61974

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
GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_599612
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
?u
?

D__inference_CNN_Model_layer_call_and_return_conditional_losses_60238

inputs
conv1_60138
conv1_60140
bn_conv1_60143
bn_conv1_60145
bn_conv1_60147
bn_conv1_60149
conv2_60154
conv2_60156
bn_conv2_60159
bn_conv2_60161
bn_conv2_60163
bn_conv2_60165
conv3_60171
conv3_60173
bn_conv3_60176
bn_conv3_60178
bn_conv3_60180
bn_conv3_60182
conv4_60188
conv4_60190
bn_conv4_60193
bn_conv4_60195
bn_conv4_60197
bn_conv4_60199
conv5_60205
conv5_60207
bn_conv5_60210
bn_conv5_60212
bn_conv5_60214
bn_conv5_60216
	fc1_60221
	fc1_60223
	fc2_60227
	fc2_60229
	fc3_60232
	fc3_60234
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall? bn_conv5/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv5/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_60138conv1_60140*
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
@__inference_conv1_layer_call_and_return_conditional_losses_592592
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_60143bn_conv1_60145bn_conv1_60147bn_conv1_60149*
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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_592942"
 bn_conv1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_593532
activation_10/PartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_587872!
max_pooling2d_8/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2_60154conv2_60156*
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
@__inference_conv2_layer_call_and_return_conditional_losses_593722
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_60159bn_conv2_60161bn_conv2_60163bn_conv2_60165*
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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_594072"
 bn_conv2/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_594662
activation_11/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_589032!
max_pooling2d_9/PartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_594872#
!dropout_8/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv3_60171conv3_60173*
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
@__inference_conv3_layer_call_and_return_conditional_losses_595152
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_60176bn_conv3_60178bn_conv3_60180bn_conv3_60182*
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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_595502"
 bn_conv3/StatefulPartitionedCall?
activation_12/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_596092
activation_12/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_590192"
 max_pooling2d_10/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
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
D__inference_dropout_9_layer_call_and_return_conditional_losses_596302#
!dropout_9/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0conv4_60188conv4_60190*
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
@__inference_conv4_layer_call_and_return_conditional_losses_596582
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_60193bn_conv4_60195bn_conv4_60197bn_conv4_60199*
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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_596932"
 bn_conv4/StatefulPartitionedCall?
activation_13/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_597522
activation_13/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_591352"
 max_pooling2d_11/PartitionedCall?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_597732$
"dropout_10/StatefulPartitionedCall?
conv5/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0conv5_60205conv5_60207*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_598012
conv5/StatefulPartitionedCall?
 bn_conv5/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0bn_conv5_60210bn_conv5_60212bn_conv5_60214bn_conv5_60216*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_598362"
 bn_conv5/StatefulPartitionedCall?
activation_14/PartitionedCallPartitionedCall)bn_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_598952
activation_14/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_599092
flatten_2/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0	fc1_60221	fc1_60223*
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
>__inference_fc1_layer_call_and_return_conditional_losses_599282
fc1/StatefulPartitionedCall?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
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
GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_599562$
"dropout_11/StatefulPartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0	fc2_60227	fc2_60229*
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
>__inference_fc2_layer_call_and_return_conditional_losses_599852
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_60232	fc3_60234*
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
>__inference_fc3_layer_call_and_return_conditional_losses_600122
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall!^bn_conv5/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2D
 bn_conv5/StatefulPartitionedCall bn_conv5/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
E
)__inference_flatten_2_layer_call_fn_61927

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_599092
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_8_layer_call_fn_58793

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
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_587872
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
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61512

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
(__inference_bn_conv1_layer_call_fn_61120

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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_587392
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
?
?
(__inference_bn_conv3_layer_call_fn_61461

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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_589712
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
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_59294

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
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_58886

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
?
I
-__inference_activation_11_layer_call_fn_61364

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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_594662
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
?
L
0__inference_max_pooling2d_10_layer_call_fn_59025

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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_590192
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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61264

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
(__inference_bn_conv3_layer_call_fn_61474

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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_590022
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
?
?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_59550

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
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_59312

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
?
I
-__inference_activation_10_layer_call_fn_61207

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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_593532
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
?
@__inference_conv1_layer_call_and_return_conditional_losses_61060

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
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_59407

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
?
d
H__inference_activation_10_layer_call_and_return_conditional_losses_61202

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
>__inference_fc2_layer_call_and_return_conditional_losses_59985

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
?
?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61816

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_10_layer_call_and_return_conditional_losses_59778

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60756

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
3bn_conv4_fusedbatchnormv3_readvariableop_1_resource(
$conv5_conv2d_readvariableop_resource)
%conv5_biasadd_readvariableop_resource$
 bn_conv5_readvariableop_resource&
"bn_conv5_readvariableop_1_resource5
1bn_conv5_fusedbatchnormv3_readvariableop_resource7
3bn_conv5_fusedbatchnormv3_readvariableop_1_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource&
"fc2_matmul_readvariableop_resource'
#fc2_biasadd_readvariableop_resource&
"fc3_matmul_readvariableop_resource'
#fc3_biasadd_readvariableop_resource
identity??bn_conv1/AssignNewValue?bn_conv1/AssignNewValue_1?(bn_conv1/FusedBatchNormV3/ReadVariableOp?*bn_conv1/FusedBatchNormV3/ReadVariableOp_1?bn_conv1/ReadVariableOp?bn_conv1/ReadVariableOp_1?bn_conv2/AssignNewValue?bn_conv2/AssignNewValue_1?(bn_conv2/FusedBatchNormV3/ReadVariableOp?*bn_conv2/FusedBatchNormV3/ReadVariableOp_1?bn_conv2/ReadVariableOp?bn_conv2/ReadVariableOp_1?bn_conv3/AssignNewValue?bn_conv3/AssignNewValue_1?(bn_conv3/FusedBatchNormV3/ReadVariableOp?*bn_conv3/FusedBatchNormV3/ReadVariableOp_1?bn_conv3/ReadVariableOp?bn_conv3/ReadVariableOp_1?bn_conv4/AssignNewValue?bn_conv4/AssignNewValue_1?(bn_conv4/FusedBatchNormV3/ReadVariableOp?*bn_conv4/FusedBatchNormV3/ReadVariableOp_1?bn_conv4/ReadVariableOp?bn_conv4/ReadVariableOp_1?bn_conv5/AssignNewValue?bn_conv5/AssignNewValue_1?(bn_conv5/FusedBatchNormV3/ReadVariableOp?*bn_conv5/FusedBatchNormV3/ReadVariableOp_1?bn_conv5/ReadVariableOp?bn_conv5/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?conv5/BiasAdd/ReadVariableOp?conv5/Conv2D/ReadVariableOp?fc1/BiasAdd/ReadVariableOp?fc1/MatMul/ReadVariableOp?fc2/BiasAdd/ReadVariableOp?fc2/MatMul/ReadVariableOp?fc3/BiasAdd/ReadVariableOp?fc3/MatMul/ReadVariableOp?
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
activation_10/ReluRelubn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????<<@2
activation_10/Relu?
max_pooling2d_8/MaxPoolMaxPool activation_10/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPool?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2D max_pooling2d_8/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
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
activation_11/ReluRelubn_conv2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
activation_11/Relu?
max_pooling2d_9/MaxPoolMaxPool activation_11/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPoolw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMul max_pooling2d_9/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeShape max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_8/dropout/Mul_1?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Ddropout_8/dropout/Mul_1:z:0#conv3/Conv2D/ReadVariableOp:value:0*
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
activation_12/ReluRelubn_conv3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_12/Relu?
max_pooling2d_10/MaxPoolMaxPool activation_12/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPoolw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul!max_pooling2d_10/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape!max_pooling2d_10/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_9/dropout/Mul_1?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4/Conv2D/ReadVariableOp?
conv4/Conv2DConv2Ddropout_9/dropout/Mul_1:z:0#conv4/Conv2D/ReadVariableOp:value:0*
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
activation_13/ReluRelubn_conv4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_13/Relu?
max_pooling2d_11/MaxPoolMaxPool activation_13/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPooly
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_10/dropout/Const?
dropout_10/dropout/MulMul!max_pooling2d_11/MaxPool:output:0!dropout_10/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_10/dropout/Mul?
dropout_10/dropout/ShapeShape!max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform?
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_10/dropout/GreaterEqual/y?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2!
dropout_10/dropout/GreaterEqual?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_10/dropout/Cast?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_10/dropout/Mul_1?
conv5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv5/Conv2D/ReadVariableOp?
conv5/Conv2DConv2Ddropout_10/dropout/Mul_1:z:0#conv5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv5/Conv2D?
conv5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv5/BiasAdd/ReadVariableOp?
conv5/BiasAddBiasAddconv5/Conv2D:output:0$conv5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv5/BiasAdd?
bn_conv5/ReadVariableOpReadVariableOp bn_conv5_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_conv5/ReadVariableOp?
bn_conv5/ReadVariableOp_1ReadVariableOp"bn_conv5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_conv5/ReadVariableOp_1?
(bn_conv5/FusedBatchNormV3/ReadVariableOpReadVariableOp1bn_conv5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(bn_conv5/FusedBatchNormV3/ReadVariableOp?
*bn_conv5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3bn_conv5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02,
*bn_conv5/FusedBatchNormV3/ReadVariableOp_1?
bn_conv5/FusedBatchNormV3FusedBatchNormV3conv5/BiasAdd:output:0bn_conv5/ReadVariableOp:value:0!bn_conv5/ReadVariableOp_1:value:00bn_conv5/FusedBatchNormV3/ReadVariableOp:value:02bn_conv5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn_conv5/FusedBatchNormV3?
bn_conv5/AssignNewValueAssignVariableOp1bn_conv5_fusedbatchnormv3_readvariableop_resource&bn_conv5/FusedBatchNormV3:batch_mean:0)^bn_conv5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*D
_class:
86loc:@bn_conv5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn_conv5/AssignNewValue?
bn_conv5/AssignNewValue_1AssignVariableOp3bn_conv5_fusedbatchnormv3_readvariableop_1_resource*bn_conv5/FusedBatchNormV3:batch_variance:0+^bn_conv5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*F
_class<
:8loc:@bn_conv5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn_conv5/AssignNewValue_1?
activation_14/ReluRelubn_conv5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
activation_14/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_2/Const?
flatten_2/ReshapeReshape activation_14/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
fc1/MatMul/ReadVariableOp?

fc1/MatMulMatMulflatten_2/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
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

fc1/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_11/dropout/Const?
dropout_11/dropout/MulMulfc1/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_11/dropout/Mulz
dropout_11/dropout/ShapeShapefc1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform?
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_11/dropout/GreaterEqual/y?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_11/dropout/GreaterEqual?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_11/dropout/Cast?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_11/dropout/Mul_1?
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
fc2/MatMul/ReadVariableOp?

fc2/MatMulMatMuldropout_11/dropout/Mul_1:z:0!fc2/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityfc3/Softmax:softmax:0^bn_conv1/AssignNewValue^bn_conv1/AssignNewValue_1)^bn_conv1/FusedBatchNormV3/ReadVariableOp+^bn_conv1/FusedBatchNormV3/ReadVariableOp_1^bn_conv1/ReadVariableOp^bn_conv1/ReadVariableOp_1^bn_conv2/AssignNewValue^bn_conv2/AssignNewValue_1)^bn_conv2/FusedBatchNormV3/ReadVariableOp+^bn_conv2/FusedBatchNormV3/ReadVariableOp_1^bn_conv2/ReadVariableOp^bn_conv2/ReadVariableOp_1^bn_conv3/AssignNewValue^bn_conv3/AssignNewValue_1)^bn_conv3/FusedBatchNormV3/ReadVariableOp+^bn_conv3/FusedBatchNormV3/ReadVariableOp_1^bn_conv3/ReadVariableOp^bn_conv3/ReadVariableOp_1^bn_conv4/AssignNewValue^bn_conv4/AssignNewValue_1)^bn_conv4/FusedBatchNormV3/ReadVariableOp+^bn_conv4/FusedBatchNormV3/ReadVariableOp_1^bn_conv4/ReadVariableOp^bn_conv4/ReadVariableOp_1^bn_conv5/AssignNewValue^bn_conv5/AssignNewValue_1)^bn_conv5/FusedBatchNormV3/ReadVariableOp+^bn_conv5/FusedBatchNormV3/ReadVariableOp_1^bn_conv5/ReadVariableOp^bn_conv5/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^conv5/BiasAdd/ReadVariableOp^conv5/Conv2D/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp^fc3/BiasAdd/ReadVariableOp^fc3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::22
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
bn_conv4/ReadVariableOp_1bn_conv4/ReadVariableOp_122
bn_conv5/AssignNewValuebn_conv5/AssignNewValue26
bn_conv5/AssignNewValue_1bn_conv5/AssignNewValue_12T
(bn_conv5/FusedBatchNormV3/ReadVariableOp(bn_conv5/FusedBatchNormV3/ReadVariableOp2X
*bn_conv5/FusedBatchNormV3/ReadVariableOp_1*bn_conv5/FusedBatchNormV3/ReadVariableOp_122
bn_conv5/ReadVariableOpbn_conv5/ReadVariableOp26
bn_conv5/ReadVariableOp_1bn_conv5/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2<
conv5/BiasAdd/ReadVariableOpconv5/BiasAdd/ReadVariableOp2:
conv5/Conv2D/ReadVariableOpconv5/Conv2D/ReadVariableOp28
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
?
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_58903

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
d
H__inference_activation_14_layer_call_and_return_conditional_losses_61911

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_59693

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
?
?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_59836

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_61376

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
?	
?
@__inference_conv1_layer_call_and_return_conditional_losses_59259

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
?
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_59630

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
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_59635

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
?
K
/__inference_max_pooling2d_9_layer_call_fn_58909

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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_589032
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
?o
?	
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60418

inputs
conv1_60318
conv1_60320
bn_conv1_60323
bn_conv1_60325
bn_conv1_60327
bn_conv1_60329
conv2_60334
conv2_60336
bn_conv2_60339
bn_conv2_60341
bn_conv2_60343
bn_conv2_60345
conv3_60351
conv3_60353
bn_conv3_60356
bn_conv3_60358
bn_conv3_60360
bn_conv3_60362
conv4_60368
conv4_60370
bn_conv4_60373
bn_conv4_60375
bn_conv4_60377
bn_conv4_60379
conv5_60385
conv5_60387
bn_conv5_60390
bn_conv5_60392
bn_conv5_60394
bn_conv5_60396
	fc1_60401
	fc1_60403
	fc2_60407
	fc2_60409
	fc3_60412
	fc3_60414
identity?? bn_conv1/StatefulPartitionedCall? bn_conv2/StatefulPartitionedCall? bn_conv3/StatefulPartitionedCall? bn_conv4/StatefulPartitionedCall? bn_conv5/StatefulPartitionedCall?conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv5/StatefulPartitionedCall?fc1/StatefulPartitionedCall?fc2/StatefulPartitionedCall?fc3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_60318conv1_60320*
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
@__inference_conv1_layer_call_and_return_conditional_losses_592592
conv1/StatefulPartitionedCall?
 bn_conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0bn_conv1_60323bn_conv1_60325bn_conv1_60327bn_conv1_60329*
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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_593122"
 bn_conv1/StatefulPartitionedCall?
activation_10/PartitionedCallPartitionedCall)bn_conv1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_10_layer_call_and_return_conditional_losses_593532
activation_10/PartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
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
GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_587872!
max_pooling2d_8/PartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2_60334conv2_60336*
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
@__inference_conv2_layer_call_and_return_conditional_losses_593722
conv2/StatefulPartitionedCall?
 bn_conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0bn_conv2_60339bn_conv2_60341bn_conv2_60343bn_conv2_60345*
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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_594252"
 bn_conv2/StatefulPartitionedCall?
activation_11/PartitionedCallPartitionedCall)bn_conv2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_11_layer_call_and_return_conditional_losses_594662
activation_11/PartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_589032!
max_pooling2d_9/PartitionedCall?
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
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
GPU2*0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_594922
dropout_8/PartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv3_60351conv3_60353*
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
@__inference_conv3_layer_call_and_return_conditional_losses_595152
conv3/StatefulPartitionedCall?
 bn_conv3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0bn_conv3_60356bn_conv3_60358bn_conv3_60360bn_conv3_60362*
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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_595682"
 bn_conv3/StatefulPartitionedCall?
activation_12/PartitionedCallPartitionedCall)bn_conv3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_12_layer_call_and_return_conditional_losses_596092
activation_12/PartitionedCall?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
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
GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_590192"
 max_pooling2d_10/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*
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
D__inference_dropout_9_layer_call_and_return_conditional_losses_596352
dropout_9/PartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0conv4_60368conv4_60370*
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
@__inference_conv4_layer_call_and_return_conditional_losses_596582
conv4/StatefulPartitionedCall?
 bn_conv4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0bn_conv4_60373bn_conv4_60375bn_conv4_60377bn_conv4_60379*
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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_597112"
 bn_conv4/StatefulPartitionedCall?
activation_13/PartitionedCallPartitionedCall)bn_conv4/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *Q
fLRJ
H__inference_activation_13_layer_call_and_return_conditional_losses_597522
activation_13/PartitionedCall?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_591352"
 max_pooling2d_11/PartitionedCall?
dropout_10/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_597782
dropout_10/PartitionedCall?
conv5/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0conv5_60385conv5_60387*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv5_layer_call_and_return_conditional_losses_598012
conv5/StatefulPartitionedCall?
 bn_conv5/StatefulPartitionedCallStatefulPartitionedCall&conv5/StatefulPartitionedCall:output:0bn_conv5_60390bn_conv5_60392bn_conv5_60394bn_conv5_60396*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_598542"
 bn_conv5/StatefulPartitionedCall?
activation_14/PartitionedCallPartitionedCall)bn_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_14_layer_call_and_return_conditional_losses_598952
activation_14/PartitionedCall?
flatten_2/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_599092
flatten_2/PartitionedCall?
fc1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0	fc1_60401	fc1_60403*
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
>__inference_fc1_layer_call_and_return_conditional_losses_599282
fc1/StatefulPartitionedCall?
dropout_11/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_599612
dropout_11/PartitionedCall?
fc2/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0	fc2_60407	fc2_60409*
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
>__inference_fc2_layer_call_and_return_conditional_losses_599852
fc2/StatefulPartitionedCall?
fc3/StatefulPartitionedCallStatefulPartitionedCall$fc2/StatefulPartitionedCall:output:0	fc3_60412	fc3_60414*
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
>__inference_fc3_layer_call_and_return_conditional_losses_600122
fc3/StatefulPartitionedCall?
IdentityIdentity$fc3/StatefulPartitionedCall:output:0!^bn_conv1/StatefulPartitionedCall!^bn_conv2/StatefulPartitionedCall!^bn_conv3/StatefulPartitionedCall!^bn_conv4/StatefulPartitionedCall!^bn_conv5/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall^fc3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::2D
 bn_conv1/StatefulPartitionedCall bn_conv1/StatefulPartitionedCall2D
 bn_conv2/StatefulPartitionedCall bn_conv2/StatefulPartitionedCall2D
 bn_conv3/StatefulPartitionedCall bn_conv3/StatefulPartitionedCall2D
 bn_conv4/StatefulPartitionedCall bn_conv4/StatefulPartitionedCall2D
 bn_conv5/StatefulPartitionedCall bn_conv5/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2:
fc3/StatefulPartitionedCallfc3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_61381

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
?	
?
@__inference_conv4_layer_call_and_return_conditional_losses_61585

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
(__inference_bn_conv4_layer_call_fn_61709

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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_590872
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
?	
?
@__inference_conv5_layer_call_and_return_conditional_losses_61769

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61862

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_9_layer_call_fn_61575

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
D__inference_dropout_9_layer_call_and_return_conditional_losses_596352
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
 
_user_specified_nameinputs
?
?
)__inference_CNN_Model_layer_call_fn_61050

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

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity??StatefulPartitionedCall?
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_CNN_Model_layer_call_and_return_conditional_losses_604182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
(__inference_bn_conv5_layer_call_fn_61842

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
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_bn_conv5_layer_call_and_return_conditional_losses_592342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
x
#__inference_fc2_layer_call_fn_61994

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
>__inference_fc2_layer_call_and_return_conditional_losses_599852
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
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_59425

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
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_61565

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
??
?
 __inference__wrapped_model_58677
input_32
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
=cnn_model_bn_conv4_fusedbatchnormv3_readvariableop_1_resource2
.cnn_model_conv5_conv2d_readvariableop_resource3
/cnn_model_conv5_biasadd_readvariableop_resource.
*cnn_model_bn_conv5_readvariableop_resource0
,cnn_model_bn_conv5_readvariableop_1_resource?
;cnn_model_bn_conv5_fusedbatchnormv3_readvariableop_resourceA
=cnn_model_bn_conv5_fusedbatchnormv3_readvariableop_1_resource0
,cnn_model_fc1_matmul_readvariableop_resource1
-cnn_model_fc1_biasadd_readvariableop_resource0
,cnn_model_fc2_matmul_readvariableop_resource1
-cnn_model_fc2_biasadd_readvariableop_resource0
,cnn_model_fc3_matmul_readvariableop_resource1
-cnn_model_fc3_biasadd_readvariableop_resource
identity??2CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv1/ReadVariableOp?#CNN_Model/bn_conv1/ReadVariableOp_1?2CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv2/ReadVariableOp?#CNN_Model/bn_conv2/ReadVariableOp_1?2CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv3/ReadVariableOp?#CNN_Model/bn_conv3/ReadVariableOp_1?2CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv4/ReadVariableOp?#CNN_Model/bn_conv4/ReadVariableOp_1?2CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp?4CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp_1?!CNN_Model/bn_conv5/ReadVariableOp?#CNN_Model/bn_conv5/ReadVariableOp_1?&CNN_Model/conv1/BiasAdd/ReadVariableOp?%CNN_Model/conv1/Conv2D/ReadVariableOp?&CNN_Model/conv2/BiasAdd/ReadVariableOp?%CNN_Model/conv2/Conv2D/ReadVariableOp?&CNN_Model/conv3/BiasAdd/ReadVariableOp?%CNN_Model/conv3/Conv2D/ReadVariableOp?&CNN_Model/conv4/BiasAdd/ReadVariableOp?%CNN_Model/conv4/Conv2D/ReadVariableOp?&CNN_Model/conv5/BiasAdd/ReadVariableOp?%CNN_Model/conv5/Conv2D/ReadVariableOp?$CNN_Model/fc1/BiasAdd/ReadVariableOp?#CNN_Model/fc1/MatMul/ReadVariableOp?$CNN_Model/fc2/BiasAdd/ReadVariableOp?#CNN_Model/fc2/MatMul/ReadVariableOp?$CNN_Model/fc3/BiasAdd/ReadVariableOp?#CNN_Model/fc3/MatMul/ReadVariableOp?
%CNN_Model/conv1/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02'
%CNN_Model/conv1/Conv2D/ReadVariableOp?
CNN_Model/conv1/Conv2DConv2Dinput_3-CNN_Model/conv1/Conv2D/ReadVariableOp:value:0*
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
CNN_Model/activation_10/ReluRelu'CNN_Model/bn_conv1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????<<@2
CNN_Model/activation_10/Relu?
!CNN_Model/max_pooling2d_8/MaxPoolMaxPool*CNN_Model/activation_10/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2#
!CNN_Model/max_pooling2d_8/MaxPool?
%CNN_Model/conv2/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02'
%CNN_Model/conv2/Conv2D/ReadVariableOp?
CNN_Model/conv2/Conv2DConv2D*CNN_Model/max_pooling2d_8/MaxPool:output:0-CNN_Model/conv2/Conv2D/ReadVariableOp:value:0*
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
CNN_Model/activation_11/ReluRelu'CNN_Model/bn_conv2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@2
CNN_Model/activation_11/Relu?
!CNN_Model/max_pooling2d_9/MaxPoolMaxPool*CNN_Model/activation_11/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2#
!CNN_Model/max_pooling2d_9/MaxPool?
CNN_Model/dropout_8/IdentityIdentity*CNN_Model/max_pooling2d_9/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
CNN_Model/dropout_8/Identity?
%CNN_Model/conv3/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02'
%CNN_Model/conv3/Conv2D/ReadVariableOp?
CNN_Model/conv3/Conv2DConv2D%CNN_Model/dropout_8/Identity:output:0-CNN_Model/conv3/Conv2D/ReadVariableOp:value:0*
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
CNN_Model/activation_12/ReluRelu'CNN_Model/bn_conv3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
CNN_Model/activation_12/Relu?
"CNN_Model/max_pooling2d_10/MaxPoolMaxPool*CNN_Model/activation_12/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"CNN_Model/max_pooling2d_10/MaxPool?
CNN_Model/dropout_9/IdentityIdentity+CNN_Model/max_pooling2d_10/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
CNN_Model/dropout_9/Identity?
%CNN_Model/conv4/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02'
%CNN_Model/conv4/Conv2D/ReadVariableOp?
CNN_Model/conv4/Conv2DConv2D%CNN_Model/dropout_9/Identity:output:0-CNN_Model/conv4/Conv2D/ReadVariableOp:value:0*
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
CNN_Model/activation_13/ReluRelu'CNN_Model/bn_conv4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
CNN_Model/activation_13/Relu?
"CNN_Model/max_pooling2d_11/MaxPoolMaxPool*CNN_Model/activation_13/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2$
"CNN_Model/max_pooling2d_11/MaxPool?
CNN_Model/dropout_10/IdentityIdentity+CNN_Model/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
CNN_Model/dropout_10/Identity?
%CNN_Model/conv5/Conv2D/ReadVariableOpReadVariableOp.cnn_model_conv5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02'
%CNN_Model/conv5/Conv2D/ReadVariableOp?
CNN_Model/conv5/Conv2DConv2D&CNN_Model/dropout_10/Identity:output:0-CNN_Model/conv5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CNN_Model/conv5/Conv2D?
&CNN_Model/conv5/BiasAdd/ReadVariableOpReadVariableOp/cnn_model_conv5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&CNN_Model/conv5/BiasAdd/ReadVariableOp?
CNN_Model/conv5/BiasAddBiasAddCNN_Model/conv5/Conv2D:output:0.CNN_Model/conv5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CNN_Model/conv5/BiasAdd?
!CNN_Model/bn_conv5/ReadVariableOpReadVariableOp*cnn_model_bn_conv5_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!CNN_Model/bn_conv5/ReadVariableOp?
#CNN_Model/bn_conv5/ReadVariableOp_1ReadVariableOp,cnn_model_bn_conv5_readvariableop_1_resource*
_output_shapes	
:?*
dtype02%
#CNN_Model/bn_conv5/ReadVariableOp_1?
2CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOpReadVariableOp;cnn_model_bn_conv5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype024
2CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp?
4CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cnn_model_bn_conv5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp_1?
#CNN_Model/bn_conv5/FusedBatchNormV3FusedBatchNormV3 CNN_Model/conv5/BiasAdd:output:0)CNN_Model/bn_conv5/ReadVariableOp:value:0+CNN_Model/bn_conv5/ReadVariableOp_1:value:0:CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp:value:0<CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2%
#CNN_Model/bn_conv5/FusedBatchNormV3?
CNN_Model/activation_14/ReluRelu'CNN_Model/bn_conv5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
CNN_Model/activation_14/Relu?
CNN_Model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
CNN_Model/flatten_2/Const?
CNN_Model/flatten_2/ReshapeReshape*CNN_Model/activation_14/Relu:activations:0"CNN_Model/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
CNN_Model/flatten_2/Reshape?
#CNN_Model/fc1/MatMul/ReadVariableOpReadVariableOp,cnn_model_fc1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#CNN_Model/fc1/MatMul/ReadVariableOp?
CNN_Model/fc1/MatMulMatMul$CNN_Model/flatten_2/Reshape:output:0+CNN_Model/fc1/MatMul/ReadVariableOp:value:0*
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
CNN_Model/dropout_11/IdentityIdentity CNN_Model/fc1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
CNN_Model/dropout_11/Identity?
#CNN_Model/fc2/MatMul/ReadVariableOpReadVariableOp,cnn_model_fc2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#CNN_Model/fc2/MatMul/ReadVariableOp?
CNN_Model/fc2/MatMulMatMul&CNN_Model/dropout_11/Identity:output:0+CNN_Model/fc2/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityCNN_Model/fc3/Softmax:softmax:03^CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv1/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv1/ReadVariableOp$^CNN_Model/bn_conv1/ReadVariableOp_13^CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv2/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv2/ReadVariableOp$^CNN_Model/bn_conv2/ReadVariableOp_13^CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv3/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv3/ReadVariableOp$^CNN_Model/bn_conv3/ReadVariableOp_13^CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv4/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv4/ReadVariableOp$^CNN_Model/bn_conv4/ReadVariableOp_13^CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp5^CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp_1"^CNN_Model/bn_conv5/ReadVariableOp$^CNN_Model/bn_conv5/ReadVariableOp_1'^CNN_Model/conv1/BiasAdd/ReadVariableOp&^CNN_Model/conv1/Conv2D/ReadVariableOp'^CNN_Model/conv2/BiasAdd/ReadVariableOp&^CNN_Model/conv2/Conv2D/ReadVariableOp'^CNN_Model/conv3/BiasAdd/ReadVariableOp&^CNN_Model/conv3/Conv2D/ReadVariableOp'^CNN_Model/conv4/BiasAdd/ReadVariableOp&^CNN_Model/conv4/Conv2D/ReadVariableOp'^CNN_Model/conv5/BiasAdd/ReadVariableOp&^CNN_Model/conv5/Conv2D/ReadVariableOp%^CNN_Model/fc1/BiasAdd/ReadVariableOp$^CNN_Model/fc1/MatMul/ReadVariableOp%^CNN_Model/fc2/BiasAdd/ReadVariableOp$^CNN_Model/fc2/MatMul/ReadVariableOp%^CNN_Model/fc3/BiasAdd/ReadVariableOp$^CNN_Model/fc3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????@@::::::::::::::::::::::::::::::::::::2h
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
#CNN_Model/bn_conv4/ReadVariableOp_1#CNN_Model/bn_conv4/ReadVariableOp_12h
2CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp2CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp2l
4CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp_14CNN_Model/bn_conv5/FusedBatchNormV3/ReadVariableOp_12F
!CNN_Model/bn_conv5/ReadVariableOp!CNN_Model/bn_conv5/ReadVariableOp2J
#CNN_Model/bn_conv5/ReadVariableOp_1#CNN_Model/bn_conv5/ReadVariableOp_12P
&CNN_Model/conv1/BiasAdd/ReadVariableOp&CNN_Model/conv1/BiasAdd/ReadVariableOp2N
%CNN_Model/conv1/Conv2D/ReadVariableOp%CNN_Model/conv1/Conv2D/ReadVariableOp2P
&CNN_Model/conv2/BiasAdd/ReadVariableOp&CNN_Model/conv2/BiasAdd/ReadVariableOp2N
%CNN_Model/conv2/Conv2D/ReadVariableOp%CNN_Model/conv2/Conv2D/ReadVariableOp2P
&CNN_Model/conv3/BiasAdd/ReadVariableOp&CNN_Model/conv3/BiasAdd/ReadVariableOp2N
%CNN_Model/conv3/Conv2D/ReadVariableOp%CNN_Model/conv3/Conv2D/ReadVariableOp2P
&CNN_Model/conv4/BiasAdd/ReadVariableOp&CNN_Model/conv4/BiasAdd/ReadVariableOp2N
%CNN_Model/conv4/Conv2D/ReadVariableOp%CNN_Model/conv4/Conv2D/ReadVariableOp2P
&CNN_Model/conv5/BiasAdd/ReadVariableOp&CNN_Model/conv5/BiasAdd/ReadVariableOp2N
%CNN_Model/conv5/Conv2D/ReadVariableOp%CNN_Model/conv5/Conv2D/ReadVariableOp2L
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
_user_specified_name	input_3
?
b
)__inference_dropout_9_layer_call_fn_61570

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
D__inference_dropout_9_layer_call_and_return_conditional_losses_596302
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
?
d
H__inference_activation_11_layer_call_and_return_conditional_losses_61359

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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61153

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
?
?
(__inference_bn_conv1_layer_call_fn_61184

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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_592942
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
?
?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_59118

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
?
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_58787

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
d
E__inference_dropout_10_layer_call_and_return_conditional_losses_61744

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61171

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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61107

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
c
*__inference_dropout_11_layer_call_fn_61969

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
GPU2*0J 8? *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_599562
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
?
z
%__inference_conv4_layer_call_fn_61594

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
@__inference_conv4_layer_call_and_return_conditional_losses_596582
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
?
?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61310

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
?
F
*__inference_dropout_10_layer_call_fn_61759

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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_10_layer_call_and_return_conditional_losses_597782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_bn_conv1_layer_call_fn_61133

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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_587702
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
?
?
(__inference_bn_conv2_layer_call_fn_61354

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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_594252
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
??
?*
__inference__traced_save_62325
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
3savev2_bn_conv4_moving_variance_read_readvariableop+
'savev2_conv5_kernel_read_readvariableop)
%savev2_conv5_bias_read_readvariableop-
)savev2_bn_conv5_gamma_read_readvariableop,
(savev2_bn_conv5_beta_read_readvariableop3
/savev2_bn_conv5_moving_mean_read_readvariableop7
3savev2_bn_conv5_moving_variance_read_readvariableop)
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
<savev2_adadelta_bn_conv4_beta_accum_grad_read_readvariableop?
;savev2_adadelta_conv5_kernel_accum_grad_read_readvariableop=
9savev2_adadelta_conv5_bias_accum_grad_read_readvariableopA
=savev2_adadelta_bn_conv5_gamma_accum_grad_read_readvariableop@
<savev2_adadelta_bn_conv5_beta_accum_grad_read_readvariableop=
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
;savev2_adadelta_bn_conv4_beta_accum_var_read_readvariableop>
:savev2_adadelta_conv5_kernel_accum_var_read_readvariableop<
8savev2_adadelta_conv5_bias_accum_var_read_readvariableop@
<savev2_adadelta_bn_conv5_gamma_accum_var_read_readvariableop?
;savev2_adadelta_bn_conv5_beta_accum_var_read_readvariableop<
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
ShardedFilename?9
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?8
value?8B?8aB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?
value?B?aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop)savev2_bn_conv1_gamma_read_readvariableop(savev2_bn_conv1_beta_read_readvariableop/savev2_bn_conv1_moving_mean_read_readvariableop3savev2_bn_conv1_moving_variance_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop)savev2_bn_conv2_gamma_read_readvariableop(savev2_bn_conv2_beta_read_readvariableop/savev2_bn_conv2_moving_mean_read_readvariableop3savev2_bn_conv2_moving_variance_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop)savev2_bn_conv3_gamma_read_readvariableop(savev2_bn_conv3_beta_read_readvariableop/savev2_bn_conv3_moving_mean_read_readvariableop3savev2_bn_conv3_moving_variance_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop)savev2_bn_conv4_gamma_read_readvariableop(savev2_bn_conv4_beta_read_readvariableop/savev2_bn_conv4_moving_mean_read_readvariableop3savev2_bn_conv4_moving_variance_read_readvariableop'savev2_conv5_kernel_read_readvariableop%savev2_conv5_bias_read_readvariableop)savev2_bn_conv5_gamma_read_readvariableop(savev2_bn_conv5_beta_read_readvariableop/savev2_bn_conv5_moving_mean_read_readvariableop3savev2_bn_conv5_moving_variance_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop%savev2_fc2_kernel_read_readvariableop#savev2_fc2_bias_read_readvariableop%savev2_fc3_kernel_read_readvariableop#savev2_fc3_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_adadelta_conv1_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv1_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv1_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv1_beta_accum_grad_read_readvariableop;savev2_adadelta_conv2_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv2_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv2_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv2_beta_accum_grad_read_readvariableop;savev2_adadelta_conv3_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv3_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv3_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv3_beta_accum_grad_read_readvariableop;savev2_adadelta_conv4_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv4_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv4_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv4_beta_accum_grad_read_readvariableop;savev2_adadelta_conv5_kernel_accum_grad_read_readvariableop9savev2_adadelta_conv5_bias_accum_grad_read_readvariableop=savev2_adadelta_bn_conv5_gamma_accum_grad_read_readvariableop<savev2_adadelta_bn_conv5_beta_accum_grad_read_readvariableop9savev2_adadelta_fc1_kernel_accum_grad_read_readvariableop7savev2_adadelta_fc1_bias_accum_grad_read_readvariableop9savev2_adadelta_fc2_kernel_accum_grad_read_readvariableop7savev2_adadelta_fc2_bias_accum_grad_read_readvariableop9savev2_adadelta_fc3_kernel_accum_grad_read_readvariableop7savev2_adadelta_fc3_bias_accum_grad_read_readvariableop:savev2_adadelta_conv1_kernel_accum_var_read_readvariableop8savev2_adadelta_conv1_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv1_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv1_beta_accum_var_read_readvariableop:savev2_adadelta_conv2_kernel_accum_var_read_readvariableop8savev2_adadelta_conv2_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv2_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv2_beta_accum_var_read_readvariableop:savev2_adadelta_conv3_kernel_accum_var_read_readvariableop8savev2_adadelta_conv3_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv3_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv3_beta_accum_var_read_readvariableop:savev2_adadelta_conv4_kernel_accum_var_read_readvariableop8savev2_adadelta_conv4_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv4_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv4_beta_accum_var_read_readvariableop:savev2_adadelta_conv5_kernel_accum_var_read_readvariableop8savev2_adadelta_conv5_bias_accum_var_read_readvariableop<savev2_adadelta_bn_conv5_gamma_accum_var_read_readvariableop;savev2_adadelta_bn_conv5_beta_accum_var_read_readvariableop8savev2_adadelta_fc1_kernel_accum_var_read_readvariableop6savev2_adadelta_fc1_bias_accum_var_read_readvariableop8savev2_adadelta_fc2_kernel_accum_var_read_readvariableop6savev2_adadelta_fc2_bias_accum_var_read_readvariableop8savev2_adadelta_fc3_kernel_accum_var_read_readvariableop6savev2_adadelta_fc3_bias_accum_var_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *o
dtypese
c2a	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:@:@:@@:@:@:@:@:@:@?:?:?:?:?:?:??:?:?:?:?:?:??:?:?:?:?:?:
??:?:
??:?:	?:: : : : : : : : :@:@:@:@:@@:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:
??:?:
??:?:	?::@:@:@:@:@@:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:
??:?:
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
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:%#!

_output_shapes
:	?: $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :,-(
&
_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@@: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@:-5)
'
_output_shapes
:@?:!6

_output_shapes	
:?:!7

_output_shapes	
:?:!8

_output_shapes	
:?:.9*
(
_output_shapes
:??:!:

_output_shapes	
:?:!;

_output_shapes	
:?:!<

_output_shapes	
:?:.=*
(
_output_shapes
:??:!>

_output_shapes	
:?:!?

_output_shapes	
:?:!@

_output_shapes	
:?:&A"
 
_output_shapes
:
??:!B

_output_shapes	
:?:&C"
 
_output_shapes
:
??:!D

_output_shapes	
:?:%E!

_output_shapes
:	?: F

_output_shapes
::,G(
&
_output_shapes
:@: H

_output_shapes
:@: I

_output_shapes
:@: J

_output_shapes
:@:,K(
&
_output_shapes
:@@: L

_output_shapes
:@: M

_output_shapes
:@: N

_output_shapes
:@:-O)
'
_output_shapes
:@?:!P

_output_shapes	
:?:!Q

_output_shapes	
:?:!R

_output_shapes	
:?:.S*
(
_output_shapes
:??:!T

_output_shapes	
:?:!U

_output_shapes	
:?:!V

_output_shapes	
:?:.W*
(
_output_shapes
:??:!X

_output_shapes	
:?:!Y

_output_shapes	
:?:!Z

_output_shapes	
:?:&["
 
_output_shapes
:
??:!\

_output_shapes	
:?:&]"
 
_output_shapes
:
??:!^

_output_shapes	
:?:%_!

_output_shapes
:	?: `

_output_shapes
::a

_output_shapes
: 
?	
?
>__inference_fc1_layer_call_and_return_conditional_losses_59928

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
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
input_38
serving_default_input_3:0?????????@@7
fc30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer-25
layer_with_weights-11
layer-26
layer_with_weights-12
layer-27
	optimizer
trainable_variables
regularization_losses
 	variables
!	keras_api
"
signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "CNN_Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["bn_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_8", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["max_pooling2d_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["bn_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_9", "inbound_nodes": [[["activation_11", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["max_pooling2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["dropout_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv3", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_12", "inbound_nodes": [[["bn_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_10", "inbound_nodes": [[["activation_12", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_9", "inbound_nodes": [[["max_pooling2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["dropout_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv4", "inbound_nodes": [[["conv4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_13", "inbound_nodes": [[["bn_conv4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_11", "inbound_nodes": [[["activation_13", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["max_pooling2d_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv5", "inbound_nodes": [[["conv5", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_14", "inbound_nodes": [[["bn_conv5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["activation_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc2", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc3", "inbound_nodes": [[["fc2", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["fc3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_10", "inbound_nodes": [[["bn_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_8", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["max_pooling2d_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_11", "inbound_nodes": [[["bn_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_9", "inbound_nodes": [[["activation_11", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_8", "inbound_nodes": [[["max_pooling2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["dropout_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv3", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_12", "inbound_nodes": [[["bn_conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_10", "inbound_nodes": [[["activation_12", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_9", "inbound_nodes": [[["max_pooling2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["dropout_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv4", "inbound_nodes": [[["conv4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_13", "inbound_nodes": [[["bn_conv4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_11", "inbound_nodes": [[["activation_13", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["max_pooling2d_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_conv5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_conv5", "inbound_nodes": [[["conv5", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_14", "inbound_nodes": [[["bn_conv5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["activation_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc1", "inbound_nodes": [[["flatten_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["fc1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc2", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc3", "inbound_nodes": [[["fc2", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["fc3", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.029999999329447746, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?	

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?	
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.	variables
/trainable_variables
0regularization_losses
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 60, 64]}}
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 64]}}
?	
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 64]}}
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?	

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}
?	
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
atrainable_variables
bregularization_losses
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 128]}}
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?	

pkernel
qbias
r	variables
strainable_variables
tregularization_losses
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 128]}}
?	
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{	variables
|trainable_variables
}regularization_losses
~	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 128]}}
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?	
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 128]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn_conv5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_conv5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 256]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fc3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc3", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
	?iter

?decay
?learning_rate
?rho#
accum_grad?$
accum_grad?*
accum_grad?+
accum_grad?:
accum_grad?;
accum_grad?A
accum_grad?B
accum_grad?U
accum_grad?V
accum_grad?\
accum_grad?]
accum_grad?p
accum_grad?q
accum_grad?w
accum_grad?x
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad??
accum_grad?#	accum_var?$	accum_var?*	accum_var?+	accum_var?:	accum_var?;	accum_var?A	accum_var?B	accum_var?U	accum_var?V	accum_var?\	accum_var?]	accum_var?p	accum_var?q	accum_var?w	accum_var?x	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var??	accum_var?"
	optimizer
?
#0
$1
*2
+3
:4
;5
A6
B7
U8
V9
\10
]11
p12
q13
w14
x15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
#0
$1
*2
+3
,4
-5
:6
;7
A8
B9
C10
D11
U12
V13
\14
]15
^16
_17
p18
q19
w20
x21
y22
z23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35"
trackable_list_wrapper
?
?layers
?metrics
?layer_metrics
 ?layer_regularization_losses
trainable_variables
regularization_losses
?non_trainable_variables
 	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$@2conv1/kernel
:@2
conv1/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
%	variables
 ?layer_regularization_losses
&trainable_variables
'regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2bn_conv1/gamma
:@2bn_conv1/beta
$:"@ (2bn_conv1/moving_mean
(:&@ (2bn_conv1/moving_variance
<
*0
+1
,2
-3"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
.	variables
 ?layer_regularization_losses
/trainable_variables
0regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
2	variables
 ?layer_regularization_losses
3trainable_variables
4regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
6	variables
 ?layer_regularization_losses
7trainable_variables
8regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv2/kernel
:@2
conv2/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
<	variables
 ?layer_regularization_losses
=trainable_variables
>regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2bn_conv2/gamma
:@2bn_conv2/beta
$:"@ (2bn_conv2/moving_mean
(:&@ (2bn_conv2/moving_variance
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
E	variables
 ?layer_regularization_losses
Ftrainable_variables
Gregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
I	variables
 ?layer_regularization_losses
Jtrainable_variables
Kregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
M	variables
 ?layer_regularization_losses
Ntrainable_variables
Oregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
Q	variables
 ?layer_regularization_losses
Rtrainable_variables
Sregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@?2conv3/kernel
:?2
conv3/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
W	variables
 ?layer_regularization_losses
Xtrainable_variables
Yregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2bn_conv3/gamma
:?2bn_conv3/beta
%:#? (2bn_conv3/moving_mean
):'? (2bn_conv3/moving_variance
<
\0
]1
^2
_3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
`	variables
 ?layer_regularization_losses
atrainable_variables
bregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
d	variables
 ?layer_regularization_losses
etrainable_variables
fregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
h	variables
 ?layer_regularization_losses
itrainable_variables
jregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
l	variables
 ?layer_regularization_losses
mtrainable_variables
nregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv4/kernel
:?2
conv4/bias
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
r	variables
 ?layer_regularization_losses
strainable_variables
tregularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2bn_conv4/gamma
:?2bn_conv4/beta
%:#? (2bn_conv4/moving_mean
):'? (2bn_conv4/moving_variance
<
w0
x1
y2
z3"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
{	variables
 ?layer_regularization_losses
|trainable_variables
}regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv5/kernel
:?2
conv5/bias
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
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2bn_conv5/gamma
:?2bn_conv5/beta
%:#? (2bn_conv5/moving_mean
):'? (2bn_conv5/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2
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
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?metrics
?	variables
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
22
23
24
25
26
27"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
h
,0
-1
C2
D3
^4
_5
y6
z7
?8
?9"
trackable_list_wrapper
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
,0
-1"
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
C0
D1"
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
^0
_1"
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
y0
z1"
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
0
?0
?1"
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
::8??2 Adadelta/conv5/kernel/accum_grad
+:)?2Adadelta/conv5/bias/accum_grad
/:-?2"Adadelta/bn_conv5/gamma/accum_grad
.:,?2!Adadelta/bn_conv5/beta/accum_grad
0:.
??2Adadelta/fc1/kernel/accum_grad
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
9:7??2Adadelta/conv5/kernel/accum_var
*:(?2Adadelta/conv5/bias/accum_var
.:,?2!Adadelta/bn_conv5/gamma/accum_var
-:+?2 Adadelta/bn_conv5/beta/accum_var
/:-
??2Adadelta/fc1/kernel/accum_var
(:&?2Adadelta/fc1/bias/accum_var
/:-
??2Adadelta/fc2/kernel/accum_var
(:&?2Adadelta/fc2/bias/accum_var
.:,	?2Adadelta/fc3/kernel/accum_var
':%2Adadelta/fc3/bias/accum_var
?2?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60756
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60029
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60896
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60132?
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
 __inference__wrapped_model_58677?
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
input_3?????????@@
?2?
)__inference_CNN_Model_layer_call_fn_61050
)__inference_CNN_Model_layer_call_fn_60493
)__inference_CNN_Model_layer_call_fn_60973
)__inference_CNN_Model_layer_call_fn_60313?
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
@__inference_conv1_layer_call_and_return_conditional_losses_61060?
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
%__inference_conv1_layer_call_fn_61069?
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
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61089
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61171
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61153
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61107?
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
(__inference_bn_conv1_layer_call_fn_61197
(__inference_bn_conv1_layer_call_fn_61120
(__inference_bn_conv1_layer_call_fn_61133
(__inference_bn_conv1_layer_call_fn_61184?
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
H__inference_activation_10_layer_call_and_return_conditional_losses_61202?
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
-__inference_activation_10_layer_call_fn_61207?
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
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_58787?
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
/__inference_max_pooling2d_8_layer_call_fn_58793?
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
@__inference_conv2_layer_call_and_return_conditional_losses_61217?
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
%__inference_conv2_layer_call_fn_61226?
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
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61328
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61246
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61310
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61264?
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
(__inference_bn_conv2_layer_call_fn_61290
(__inference_bn_conv2_layer_call_fn_61354
(__inference_bn_conv2_layer_call_fn_61341
(__inference_bn_conv2_layer_call_fn_61277?
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
H__inference_activation_11_layer_call_and_return_conditional_losses_61359?
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
-__inference_activation_11_layer_call_fn_61364?
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
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_58903?
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
/__inference_max_pooling2d_9_layer_call_fn_58909?
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
D__inference_dropout_8_layer_call_and_return_conditional_losses_61376
D__inference_dropout_8_layer_call_and_return_conditional_losses_61381?
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
)__inference_dropout_8_layer_call_fn_61391
)__inference_dropout_8_layer_call_fn_61386?
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
@__inference_conv3_layer_call_and_return_conditional_losses_61401?
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
%__inference_conv3_layer_call_fn_61410?
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
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61448
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61430
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61494
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61512?
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
(__inference_bn_conv3_layer_call_fn_61525
(__inference_bn_conv3_layer_call_fn_61461
(__inference_bn_conv3_layer_call_fn_61538
(__inference_bn_conv3_layer_call_fn_61474?
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
H__inference_activation_12_layer_call_and_return_conditional_losses_61543?
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
-__inference_activation_12_layer_call_fn_61548?
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
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_59019?
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
0__inference_max_pooling2d_10_layer_call_fn_59025?
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
D__inference_dropout_9_layer_call_and_return_conditional_losses_61560
D__inference_dropout_9_layer_call_and_return_conditional_losses_61565?
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
)__inference_dropout_9_layer_call_fn_61575
)__inference_dropout_9_layer_call_fn_61570?
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
@__inference_conv4_layer_call_and_return_conditional_losses_61585?
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
%__inference_conv4_layer_call_fn_61594?
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
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61696
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61632
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61614
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61678?
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
(__inference_bn_conv4_layer_call_fn_61722
(__inference_bn_conv4_layer_call_fn_61658
(__inference_bn_conv4_layer_call_fn_61645
(__inference_bn_conv4_layer_call_fn_61709?
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
H__inference_activation_13_layer_call_and_return_conditional_losses_61727?
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
-__inference_activation_13_layer_call_fn_61732?
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
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_59135?
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
0__inference_max_pooling2d_11_layer_call_fn_59141?
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
E__inference_dropout_10_layer_call_and_return_conditional_losses_61744
E__inference_dropout_10_layer_call_and_return_conditional_losses_61749?
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
*__inference_dropout_10_layer_call_fn_61759
*__inference_dropout_10_layer_call_fn_61754?
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
@__inference_conv5_layer_call_and_return_conditional_losses_61769?
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
%__inference_conv5_layer_call_fn_61778?
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
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61880
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61862
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61816
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61798?
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
(__inference_bn_conv5_layer_call_fn_61829
(__inference_bn_conv5_layer_call_fn_61842
(__inference_bn_conv5_layer_call_fn_61893
(__inference_bn_conv5_layer_call_fn_61906?
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
H__inference_activation_14_layer_call_and_return_conditional_losses_61911?
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
-__inference_activation_14_layer_call_fn_61916?
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_61922?
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
)__inference_flatten_2_layer_call_fn_61927?
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
>__inference_fc1_layer_call_and_return_conditional_losses_61938?
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
#__inference_fc1_layer_call_fn_61947?
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
E__inference_dropout_11_layer_call_and_return_conditional_losses_61959
E__inference_dropout_11_layer_call_and_return_conditional_losses_61964?
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
*__inference_dropout_11_layer_call_fn_61969
*__inference_dropout_11_layer_call_fn_61974?
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
>__inference_fc2_layer_call_and_return_conditional_losses_61985?
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
#__inference_fc2_layer_call_fn_61994?
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
>__inference_fc3_layer_call_and_return_conditional_losses_62005?
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
#__inference_fc3_layer_call_fn_62014?
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
#__inference_signature_wrapper_60578input_3"?
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
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60029?0#$*+,-:;ABCDUV\]^_pqwxyz????????????@?=
6?3
)?&
input_3?????????@@
p

 
? "%?"
?
0?????????
? ?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60132?0#$*+,-:;ABCDUV\]^_pqwxyz????????????@?=
6?3
)?&
input_3?????????@@
p 

 
? "%?"
?
0?????????
? ?
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60756?0#$*+,-:;ABCDUV\]^_pqwxyz??????????????<
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
D__inference_CNN_Model_layer_call_and_return_conditional_losses_60896?0#$*+,-:;ABCDUV\]^_pqwxyz??????????????<
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
)__inference_CNN_Model_layer_call_fn_60313?0#$*+,-:;ABCDUV\]^_pqwxyz????????????@?=
6?3
)?&
input_3?????????@@
p

 
? "???????????
)__inference_CNN_Model_layer_call_fn_60493?0#$*+,-:;ABCDUV\]^_pqwxyz????????????@?=
6?3
)?&
input_3?????????@@
p 

 
? "???????????
)__inference_CNN_Model_layer_call_fn_60973?0#$*+,-:;ABCDUV\]^_pqwxyz??????????????<
5?2
(?%
inputs?????????@@
p

 
? "???????????
)__inference_CNN_Model_layer_call_fn_61050?0#$*+,-:;ABCDUV\]^_pqwxyz??????????????<
5?2
(?%
inputs?????????@@
p 

 
? "???????????
 __inference__wrapped_model_58677?0#$*+,-:;ABCDUV\]^_pqwxyz????????????8?5
.?+
)?&
input_3?????????@@
? ")?&
$
fc3?
fc3??????????
H__inference_activation_10_layer_call_and_return_conditional_losses_61202h7?4
-?*
(?%
inputs?????????<<@
? "-?*
#? 
0?????????<<@
? ?
-__inference_activation_10_layer_call_fn_61207[7?4
-?*
(?%
inputs?????????<<@
? " ??????????<<@?
H__inference_activation_11_layer_call_and_return_conditional_losses_61359h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
-__inference_activation_11_layer_call_fn_61364[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
H__inference_activation_12_layer_call_and_return_conditional_losses_61543j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_activation_12_layer_call_fn_61548]8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_activation_13_layer_call_and_return_conditional_losses_61727j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_activation_13_layer_call_fn_61732]8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_activation_14_layer_call_and_return_conditional_losses_61911j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_activation_14_layer_call_fn_61916]8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61089?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61107?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61153r*+,-;?8
1?.
(?%
inputs?????????<<@
p
? "-?*
#? 
0?????????<<@
? ?
C__inference_bn_conv1_layer_call_and_return_conditional_losses_61171r*+,-;?8
1?.
(?%
inputs?????????<<@
p 
? "-?*
#? 
0?????????<<@
? ?
(__inference_bn_conv1_layer_call_fn_61120?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
(__inference_bn_conv1_layer_call_fn_61133?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
(__inference_bn_conv1_layer_call_fn_61184e*+,-;?8
1?.
(?%
inputs?????????<<@
p
? " ??????????<<@?
(__inference_bn_conv1_layer_call_fn_61197e*+,-;?8
1?.
(?%
inputs?????????<<@
p 
? " ??????????<<@?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61246?ABCDM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61264?ABCDM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61310rABCD;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
C__inference_bn_conv2_layer_call_and_return_conditional_losses_61328rABCD;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
(__inference_bn_conv2_layer_call_fn_61277?ABCDM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
(__inference_bn_conv2_layer_call_fn_61290?ABCDM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
(__inference_bn_conv2_layer_call_fn_61341eABCD;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
(__inference_bn_conv2_layer_call_fn_61354eABCD;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61430?\]^_N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61448?\]^_N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61494t\]^_<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
C__inference_bn_conv3_layer_call_and_return_conditional_losses_61512t\]^_<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
(__inference_bn_conv3_layer_call_fn_61461?\]^_N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
(__inference_bn_conv3_layer_call_fn_61474?\]^_N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
(__inference_bn_conv3_layer_call_fn_61525g\]^_<?9
2?/
)?&
inputs??????????
p
? "!????????????
(__inference_bn_conv3_layer_call_fn_61538g\]^_<?9
2?/
)?&
inputs??????????
p 
? "!????????????
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61614twxyz<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61632twxyz<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61678?wxyzN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv4_layer_call_and_return_conditional_losses_61696?wxyzN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_bn_conv4_layer_call_fn_61645gwxyz<?9
2?/
)?&
inputs??????????
p
? "!????????????
(__inference_bn_conv4_layer_call_fn_61658gwxyz<?9
2?/
)?&
inputs??????????
p 
? "!????????????
(__inference_bn_conv4_layer_call_fn_61709?wxyzN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
(__inference_bn_conv4_layer_call_fn_61722?wxyzN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61798?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61816?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61862x????<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
C__inference_bn_conv5_layer_call_and_return_conditional_losses_61880x????<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
(__inference_bn_conv5_layer_call_fn_61829?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
(__inference_bn_conv5_layer_call_fn_61842?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
(__inference_bn_conv5_layer_call_fn_61893k????<?9
2?/
)?&
inputs??????????
p
? "!????????????
(__inference_bn_conv5_layer_call_fn_61906k????<?9
2?/
)?&
inputs??????????
p 
? "!????????????
@__inference_conv1_layer_call_and_return_conditional_losses_61060l#$7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????<<@
? ?
%__inference_conv1_layer_call_fn_61069_#$7?4
-?*
(?%
inputs?????????@@
? " ??????????<<@?
@__inference_conv2_layer_call_and_return_conditional_losses_61217l:;7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
%__inference_conv2_layer_call_fn_61226_:;7?4
-?*
(?%
inputs?????????@
? " ??????????@?
@__inference_conv3_layer_call_and_return_conditional_losses_61401mUV7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
%__inference_conv3_layer_call_fn_61410`UV7?4
-?*
(?%
inputs?????????@
? "!????????????
@__inference_conv4_layer_call_and_return_conditional_losses_61585npq8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
%__inference_conv4_layer_call_fn_61594apq8?5
.?+
)?&
inputs??????????
? "!????????????
@__inference_conv5_layer_call_and_return_conditional_losses_61769p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
%__inference_conv5_layer_call_fn_61778c??8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_dropout_10_layer_call_and_return_conditional_losses_61744n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
E__inference_dropout_10_layer_call_and_return_conditional_losses_61749n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
*__inference_dropout_10_layer_call_fn_61754a<?9
2?/
)?&
inputs??????????
p
? "!????????????
*__inference_dropout_10_layer_call_fn_61759a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
E__inference_dropout_11_layer_call_and_return_conditional_losses_61959^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
E__inference_dropout_11_layer_call_and_return_conditional_losses_61964^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? 
*__inference_dropout_11_layer_call_fn_61969Q4?1
*?'
!?
inputs??????????
p
? "???????????
*__inference_dropout_11_layer_call_fn_61974Q4?1
*?'
!?
inputs??????????
p 
? "????????????
D__inference_dropout_8_layer_call_and_return_conditional_losses_61376l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
D__inference_dropout_8_layer_call_and_return_conditional_losses_61381l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
)__inference_dropout_8_layer_call_fn_61386_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
)__inference_dropout_8_layer_call_fn_61391_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
D__inference_dropout_9_layer_call_and_return_conditional_losses_61560n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
D__inference_dropout_9_layer_call_and_return_conditional_losses_61565n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
)__inference_dropout_9_layer_call_fn_61570a<?9
2?/
)?&
inputs??????????
p
? "!????????????
)__inference_dropout_9_layer_call_fn_61575a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
>__inference_fc1_layer_call_and_return_conditional_losses_61938`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
#__inference_fc1_layer_call_fn_61947S??0?-
&?#
!?
inputs??????????
? "????????????
>__inference_fc2_layer_call_and_return_conditional_losses_61985`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
#__inference_fc2_layer_call_fn_61994S??0?-
&?#
!?
inputs??????????
? "????????????
>__inference_fc3_layer_call_and_return_conditional_losses_62005_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
#__inference_fc3_layer_call_fn_62014R??0?-
&?#
!?
inputs??????????
? "???????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_61922b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
)__inference_flatten_2_layer_call_fn_61927U8?5
.?+
)?&
inputs??????????
? "????????????
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_59019?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_10_layer_call_fn_59025?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_59135?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_11_layer_call_fn_59141?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_58787?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_8_layer_call_fn_58793?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_58903?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_9_layer_call_fn_58909?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
#__inference_signature_wrapper_60578?0#$*+,-:;ABCDUV\]^_pqwxyz????????????C?@
? 
9?6
4
input_3)?&
input_3?????????@@")?&
$
fc3?
fc3?????????
ח
??
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
 ?"serve*2.6.22v2.6.1-9-gc2363d6d0258??	
~
dense_3820/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namedense_3820/kernel
w
%dense_3820/kernel/Read/ReadVariableOpReadVariableOpdense_3820/kernel*
_output_shapes

: *
dtype0
v
dense_3820/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedense_3820/bias
o
#dense_3820/bias/Read/ReadVariableOpReadVariableOpdense_3820/bias*
_output_shapes
: *
dtype0
~
dense_3821/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namedense_3821/kernel
w
%dense_3821/kernel/Read/ReadVariableOpReadVariableOpdense_3821/kernel*
_output_shapes

: *
dtype0
v
dense_3821/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3821/bias
o
#dense_3821/bias/Read/ReadVariableOpReadVariableOpdense_3821/bias*
_output_shapes
:*
dtype0
~
dense_3822/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3822/kernel
w
%dense_3822/kernel/Read/ReadVariableOpReadVariableOpdense_3822/kernel*
_output_shapes

:*
dtype0
v
dense_3822/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3822/bias
o
#dense_3822/bias/Read/ReadVariableOpReadVariableOpdense_3822/bias*
_output_shapes
:*
dtype0
~
dense_3823/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3823/kernel
w
%dense_3823/kernel/Read/ReadVariableOpReadVariableOpdense_3823/kernel*
_output_shapes

:*
dtype0
v
dense_3823/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3823/bias
o
#dense_3823/bias/Read/ReadVariableOpReadVariableOpdense_3823/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
Adam/dense_3820/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/dense_3820/kernel/m
?
,Adam/dense_3820/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3820/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_3820/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/dense_3820/bias/m
}
*Adam/dense_3820/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3820/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_3821/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/dense_3821/kernel/m
?
,Adam/dense_3821/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3821/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_3821/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3821/bias/m
}
*Adam/dense_3821/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3821/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3822/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_3822/kernel/m
?
,Adam/dense_3822/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3822/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_3822/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3822/bias/m
}
*Adam/dense_3822/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3822/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3823/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_3823/kernel/m
?
,Adam/dense_3823/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3823/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_3823/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3823/bias/m
}
*Adam/dense_3823/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3823/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3820/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/dense_3820/kernel/v
?
,Adam/dense_3820/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3820/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_3820/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/dense_3820/bias/v
}
*Adam/dense_3820/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3820/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_3821/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/dense_3821/kernel/v
?
,Adam/dense_3821/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3821/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_3821/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3821/bias/v
}
*Adam/dense_3821/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3821/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3822/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_3822/kernel/v
?
,Adam/dense_3822/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3822/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_3822/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3822/bias/v
}
*Adam/dense_3822/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3822/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3823/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_3823/kernel/v
?
,Adam/dense_3823/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3823/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_3823/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_3823/bias/v
}
*Adam/dense_3823/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3823/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
h

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm"mn#mo,mp-mqvrvsvtvu"vv#vw,vx-vy
8
0
1
2
3
"4
#5
,6
-7
 
8
0
1
2
3
"4
#5
,6
-7
?
7layer_metrics
		variables
8metrics

regularization_losses
trainable_variables
9non_trainable_variables
:layer_regularization_losses

;layers
 
][
VARIABLE_VALUEdense_3820/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3820/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
<layer_metrics
	variables
=metrics
regularization_losses
trainable_variables
>non_trainable_variables
?layer_regularization_losses

@layers
 
 
 
?
Alayer_metrics
	variables
Bmetrics
regularization_losses
trainable_variables
Cnon_trainable_variables
Dlayer_regularization_losses

Elayers
][
VARIABLE_VALUEdense_3821/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3821/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Flayer_metrics
	variables
Gmetrics
regularization_losses
trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses

Jlayers
 
 
 
?
Klayer_metrics
	variables
Lmetrics
regularization_losses
 trainable_variables
Mnon_trainable_variables
Nlayer_regularization_losses

Olayers
][
VARIABLE_VALUEdense_3822/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3822/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?
Player_metrics
$	variables
Qmetrics
%regularization_losses
&trainable_variables
Rnon_trainable_variables
Slayer_regularization_losses

Tlayers
 
 
 
?
Ulayer_metrics
(	variables
Vmetrics
)regularization_losses
*trainable_variables
Wnon_trainable_variables
Xlayer_regularization_losses

Ylayers
][
VARIABLE_VALUEdense_3823/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3823/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
?
Zlayer_metrics
.	variables
[metrics
/regularization_losses
0trainable_variables
\non_trainable_variables
]layer_regularization_losses

^layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

_0
`1
 
 
1
0
1
2
3
4
5
6
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
 
 
 
 
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
?~
VARIABLE_VALUEAdam/dense_3820/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3820/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_3821/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3821/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_3822/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3822/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_3823/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3823/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_3820/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3820/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_3821/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3821/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_3822/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3822/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/dense_3823/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_3823/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputdense_3820/kerneldense_3820/biasdense_3821/kerneldense_3821/biasdense_3822/kerneldense_3822/biasdense_3823/kerneldense_3823/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_8647541
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_3820/kernel/Read/ReadVariableOp#dense_3820/bias/Read/ReadVariableOp%dense_3821/kernel/Read/ReadVariableOp#dense_3821/bias/Read/ReadVariableOp%dense_3822/kernel/Read/ReadVariableOp#dense_3822/bias/Read/ReadVariableOp%dense_3823/kernel/Read/ReadVariableOp#dense_3823/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/dense_3820/kernel/m/Read/ReadVariableOp*Adam/dense_3820/bias/m/Read/ReadVariableOp,Adam/dense_3821/kernel/m/Read/ReadVariableOp*Adam/dense_3821/bias/m/Read/ReadVariableOp,Adam/dense_3822/kernel/m/Read/ReadVariableOp*Adam/dense_3822/bias/m/Read/ReadVariableOp,Adam/dense_3823/kernel/m/Read/ReadVariableOp*Adam/dense_3823/bias/m/Read/ReadVariableOp,Adam/dense_3820/kernel/v/Read/ReadVariableOp*Adam/dense_3820/bias/v/Read/ReadVariableOp,Adam/dense_3821/kernel/v/Read/ReadVariableOp*Adam/dense_3821/bias/v/Read/ReadVariableOp,Adam/dense_3822/kernel/v/Read/ReadVariableOp*Adam/dense_3822/bias/v/Read/ReadVariableOp,Adam/dense_3823/kernel/v/Read/ReadVariableOp*Adam/dense_3823/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_8648062
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3820/kerneldense_3820/biasdense_3821/kerneldense_3821/biasdense_3822/kerneldense_3822/biasdense_3823/kerneldense_3823/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_3820/kernel/mAdam/dense_3820/bias/mAdam/dense_3821/kernel/mAdam/dense_3821/bias/mAdam/dense_3822/kernel/mAdam/dense_3822/bias/mAdam/dense_3823/kernel/mAdam/dense_3823/bias/mAdam/dense_3820/kernel/vAdam/dense_3820/bias/vAdam/dense_3821/kernel/vAdam/dense_3821/bias/vAdam/dense_3822/kernel/vAdam/dense_3822/bias/vAdam/dense_3823/kernel/vAdam/dense_3823/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_8648171??
?
?
__inference_loss_fn_0_8647918K
9dense_3820_kernel_regularizer_abs_readvariableop_resource: 
identity??0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp9dense_3820_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mulo
IdentityIdentity%dense_3820/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp
?
g
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647875

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
0__inference_sequential_928_layer_call_fn_8647187	
input
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_928_layer_call_and_return_conditional_losses_86471682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
G__inference_dense_3821_layer_call_and_return_conditional_losses_8647801

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_8647940K
9dense_3822_kernel_regularizer_abs_readvariableop_resource:
identity??0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp9dense_3822_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mulo
IdentityIdentity%dense_3822/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp
?
h
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647828

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_8647541	
input
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_86470352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
J
.__inference_dropout_2897_layer_call_fn_8647865

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2897_layer_call_and_return_conditional_losses_86471302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647449	
input$
dense_3820_8647407:  
dense_3820_8647409: $
dense_3821_8647413:  
dense_3821_8647415:$
dense_3822_8647419: 
dense_3822_8647421:$
dense_3823_8647425: 
dense_3823_8647427:
identity??"dense_3820/StatefulPartitionedCall?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?"dense_3821/StatefulPartitionedCall?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?"dense_3822/StatefulPartitionedCall?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?"dense_3823/StatefulPartitionedCall?
"dense_3820/StatefulPartitionedCallStatefulPartitionedCallinputdense_3820_8647407dense_3820_8647409*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3820_layer_call_and_return_conditional_losses_86470592$
"dense_3820/StatefulPartitionedCall?
dropout_2895/PartitionedCallPartitionedCall+dense_3820/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2895_layer_call_and_return_conditional_losses_86470702
dropout_2895/PartitionedCall?
"dense_3821/StatefulPartitionedCallStatefulPartitionedCall%dropout_2895/PartitionedCall:output:0dense_3821_8647413dense_3821_8647415*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3821_layer_call_and_return_conditional_losses_86470892$
"dense_3821/StatefulPartitionedCall?
dropout_2896/PartitionedCallPartitionedCall+dense_3821/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2896_layer_call_and_return_conditional_losses_86471002
dropout_2896/PartitionedCall?
"dense_3822/StatefulPartitionedCallStatefulPartitionedCall%dropout_2896/PartitionedCall:output:0dense_3822_8647419dense_3822_8647421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3822_layer_call_and_return_conditional_losses_86471192$
"dense_3822/StatefulPartitionedCall?
dropout_2897/PartitionedCallPartitionedCall+dense_3822/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2897_layer_call_and_return_conditional_losses_86471302
dropout_2897/PartitionedCall?
"dense_3823/StatefulPartitionedCallStatefulPartitionedCall%dropout_2897/PartitionedCall:output:0dense_3823_8647425dense_3823_8647427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3823_layer_call_and_return_conditional_losses_86471432$
"dense_3823/StatefulPartitionedCall?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3820_8647407*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mul?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3821_8647413*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mul?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3822_8647419*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mul?
IdentityIdentity+dense_3823/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^dense_3820/StatefulPartitionedCall1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp#^dense_3821/StatefulPartitionedCall1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp#^dense_3822/StatefulPartitionedCall1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp#^dense_3823/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_3820/StatefulPartitionedCall"dense_3820/StatefulPartitionedCall2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3821/StatefulPartitionedCall"dense_3821/StatefulPartitionedCall2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3822/StatefulPartitionedCall"dense_3822/StatefulPartitionedCall2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3823/StatefulPartitionedCall"dense_3823/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?;
?
"__inference__wrapped_model_8647035	
inputJ
8sequential_928_dense_3820_matmul_readvariableop_resource: G
9sequential_928_dense_3820_biasadd_readvariableop_resource: J
8sequential_928_dense_3821_matmul_readvariableop_resource: G
9sequential_928_dense_3821_biasadd_readvariableop_resource:J
8sequential_928_dense_3822_matmul_readvariableop_resource:G
9sequential_928_dense_3822_biasadd_readvariableop_resource:J
8sequential_928_dense_3823_matmul_readvariableop_resource:G
9sequential_928_dense_3823_biasadd_readvariableop_resource:
identity??0sequential_928/dense_3820/BiasAdd/ReadVariableOp?/sequential_928/dense_3820/MatMul/ReadVariableOp?0sequential_928/dense_3821/BiasAdd/ReadVariableOp?/sequential_928/dense_3821/MatMul/ReadVariableOp?0sequential_928/dense_3822/BiasAdd/ReadVariableOp?/sequential_928/dense_3822/MatMul/ReadVariableOp?0sequential_928/dense_3823/BiasAdd/ReadVariableOp?/sequential_928/dense_3823/MatMul/ReadVariableOp?
/sequential_928/dense_3820/MatMul/ReadVariableOpReadVariableOp8sequential_928_dense_3820_matmul_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_928/dense_3820/MatMul/ReadVariableOp?
 sequential_928/dense_3820/MatMulMatMulinput7sequential_928/dense_3820/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_928/dense_3820/MatMul?
0sequential_928/dense_3820/BiasAdd/ReadVariableOpReadVariableOp9sequential_928_dense_3820_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential_928/dense_3820/BiasAdd/ReadVariableOp?
!sequential_928/dense_3820/BiasAddBiasAdd*sequential_928/dense_3820/MatMul:product:08sequential_928/dense_3820/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2#
!sequential_928/dense_3820/BiasAdd?
sequential_928/dense_3820/ReluRelu*sequential_928/dense_3820/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2 
sequential_928/dense_3820/Relu?
$sequential_928/dropout_2895/IdentityIdentity,sequential_928/dense_3820/Relu:activations:0*
T0*'
_output_shapes
:????????? 2&
$sequential_928/dropout_2895/Identity?
/sequential_928/dense_3821/MatMul/ReadVariableOpReadVariableOp8sequential_928_dense_3821_matmul_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_928/dense_3821/MatMul/ReadVariableOp?
 sequential_928/dense_3821/MatMulMatMul-sequential_928/dropout_2895/Identity:output:07sequential_928/dense_3821/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_928/dense_3821/MatMul?
0sequential_928/dense_3821/BiasAdd/ReadVariableOpReadVariableOp9sequential_928_dense_3821_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential_928/dense_3821/BiasAdd/ReadVariableOp?
!sequential_928/dense_3821/BiasAddBiasAdd*sequential_928/dense_3821/MatMul:product:08sequential_928/dense_3821/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!sequential_928/dense_3821/BiasAdd?
sequential_928/dense_3821/ReluRelu*sequential_928/dense_3821/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_928/dense_3821/Relu?
$sequential_928/dropout_2896/IdentityIdentity,sequential_928/dense_3821/Relu:activations:0*
T0*'
_output_shapes
:?????????2&
$sequential_928/dropout_2896/Identity?
/sequential_928/dense_3822/MatMul/ReadVariableOpReadVariableOp8sequential_928_dense_3822_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/sequential_928/dense_3822/MatMul/ReadVariableOp?
 sequential_928/dense_3822/MatMulMatMul-sequential_928/dropout_2896/Identity:output:07sequential_928/dense_3822/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_928/dense_3822/MatMul?
0sequential_928/dense_3822/BiasAdd/ReadVariableOpReadVariableOp9sequential_928_dense_3822_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential_928/dense_3822/BiasAdd/ReadVariableOp?
!sequential_928/dense_3822/BiasAddBiasAdd*sequential_928/dense_3822/MatMul:product:08sequential_928/dense_3822/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!sequential_928/dense_3822/BiasAdd?
sequential_928/dense_3822/ReluRelu*sequential_928/dense_3822/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_928/dense_3822/Relu?
$sequential_928/dropout_2897/IdentityIdentity,sequential_928/dense_3822/Relu:activations:0*
T0*'
_output_shapes
:?????????2&
$sequential_928/dropout_2897/Identity?
/sequential_928/dense_3823/MatMul/ReadVariableOpReadVariableOp8sequential_928_dense_3823_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/sequential_928/dense_3823/MatMul/ReadVariableOp?
 sequential_928/dense_3823/MatMulMatMul-sequential_928/dropout_2897/Identity:output:07sequential_928/dense_3823/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_928/dense_3823/MatMul?
0sequential_928/dense_3823/BiasAdd/ReadVariableOpReadVariableOp9sequential_928_dense_3823_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential_928/dense_3823/BiasAdd/ReadVariableOp?
!sequential_928/dense_3823/BiasAddBiasAdd*sequential_928/dense_3823/MatMul:product:08sequential_928/dense_3823/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!sequential_928/dense_3823/BiasAdd?
!sequential_928/dense_3823/SoftmaxSoftmax*sequential_928/dense_3823/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2#
!sequential_928/dense_3823/Softmax?
IdentityIdentity+sequential_928/dense_3823/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp1^sequential_928/dense_3820/BiasAdd/ReadVariableOp0^sequential_928/dense_3820/MatMul/ReadVariableOp1^sequential_928/dense_3821/BiasAdd/ReadVariableOp0^sequential_928/dense_3821/MatMul/ReadVariableOp1^sequential_928/dense_3822/BiasAdd/ReadVariableOp0^sequential_928/dense_3822/MatMul/ReadVariableOp1^sequential_928/dense_3823/BiasAdd/ReadVariableOp0^sequential_928/dense_3823/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2d
0sequential_928/dense_3820/BiasAdd/ReadVariableOp0sequential_928/dense_3820/BiasAdd/ReadVariableOp2b
/sequential_928/dense_3820/MatMul/ReadVariableOp/sequential_928/dense_3820/MatMul/ReadVariableOp2d
0sequential_928/dense_3821/BiasAdd/ReadVariableOp0sequential_928/dense_3821/BiasAdd/ReadVariableOp2b
/sequential_928/dense_3821/MatMul/ReadVariableOp/sequential_928/dense_3821/MatMul/ReadVariableOp2d
0sequential_928/dense_3822/BiasAdd/ReadVariableOp0sequential_928/dense_3822/BiasAdd/ReadVariableOp2b
/sequential_928/dense_3822/MatMul/ReadVariableOp/sequential_928/dense_3822/MatMul/ReadVariableOp2d
0sequential_928/dense_3823/BiasAdd/ReadVariableOp0sequential_928/dense_3823/BiasAdd/ReadVariableOp2b
/sequential_928/dense_3823/MatMul/ReadVariableOp/sequential_928/dense_3823/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
,__inference_dense_3822_layer_call_fn_8647843

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3822_layer_call_and_return_conditional_losses_86471192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647070

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_dense_3823_layer_call_fn_8647896

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3823_layer_call_and_return_conditional_losses_86471432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_8647929K
9dense_3821_kernel_regularizer_abs_readvariableop_resource: 
identity??0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp9dense_3821_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mulo
IdentityIdentity%dense_3821/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp
?
g
.__inference_dropout_2895_layer_call_fn_8647752

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2895_layer_call_and_return_conditional_losses_86472832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?L
?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647636

inputs;
)dense_3820_matmul_readvariableop_resource: 8
*dense_3820_biasadd_readvariableop_resource: ;
)dense_3821_matmul_readvariableop_resource: 8
*dense_3821_biasadd_readvariableop_resource:;
)dense_3822_matmul_readvariableop_resource:8
*dense_3822_biasadd_readvariableop_resource:;
)dense_3823_matmul_readvariableop_resource:8
*dense_3823_biasadd_readvariableop_resource:
identity??!dense_3820/BiasAdd/ReadVariableOp? dense_3820/MatMul/ReadVariableOp?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?!dense_3821/BiasAdd/ReadVariableOp? dense_3821/MatMul/ReadVariableOp?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?!dense_3822/BiasAdd/ReadVariableOp? dense_3822/MatMul/ReadVariableOp?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?!dense_3823/BiasAdd/ReadVariableOp? dense_3823/MatMul/ReadVariableOp?
 dense_3820/MatMul/ReadVariableOpReadVariableOp)dense_3820_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_3820/MatMul/ReadVariableOp?
dense_3820/MatMulMatMulinputs(dense_3820/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_3820/MatMul?
!dense_3820/BiasAdd/ReadVariableOpReadVariableOp*dense_3820_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!dense_3820/BiasAdd/ReadVariableOp?
dense_3820/BiasAddBiasAdddense_3820/MatMul:product:0)dense_3820/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_3820/BiasAddy
dense_3820/ReluReludense_3820/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_3820/Relu?
dropout_2895/IdentityIdentitydense_3820/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
dropout_2895/Identity?
 dense_3821/MatMul/ReadVariableOpReadVariableOp)dense_3821_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_3821/MatMul/ReadVariableOp?
dense_3821/MatMulMatMuldropout_2895/Identity:output:0(dense_3821/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3821/MatMul?
!dense_3821/BiasAdd/ReadVariableOpReadVariableOp*dense_3821_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3821/BiasAdd/ReadVariableOp?
dense_3821/BiasAddBiasAdddense_3821/MatMul:product:0)dense_3821/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3821/BiasAddy
dense_3821/ReluReludense_3821/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3821/Relu?
dropout_2896/IdentityIdentitydense_3821/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_2896/Identity?
 dense_3822/MatMul/ReadVariableOpReadVariableOp)dense_3822_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3822/MatMul/ReadVariableOp?
dense_3822/MatMulMatMuldropout_2896/Identity:output:0(dense_3822/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3822/MatMul?
!dense_3822/BiasAdd/ReadVariableOpReadVariableOp*dense_3822_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3822/BiasAdd/ReadVariableOp?
dense_3822/BiasAddBiasAdddense_3822/MatMul:product:0)dense_3822/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3822/BiasAddy
dense_3822/ReluReludense_3822/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3822/Relu?
dropout_2897/IdentityIdentitydense_3822/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_2897/Identity?
 dense_3823/MatMul/ReadVariableOpReadVariableOp)dense_3823_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3823/MatMul/ReadVariableOp?
dense_3823/MatMulMatMuldropout_2897/Identity:output:0(dense_3823/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3823/MatMul?
!dense_3823/BiasAdd/ReadVariableOpReadVariableOp*dense_3823_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3823/BiasAdd/ReadVariableOp?
dense_3823/BiasAddBiasAdddense_3823/MatMul:product:0)dense_3823/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3823/BiasAdd?
dense_3823/SoftmaxSoftmaxdense_3823/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3823/Softmax?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp)dense_3820_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mul?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp)dense_3821_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mul?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp)dense_3822_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mulw
IdentityIdentitydense_3823/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_3820/BiasAdd/ReadVariableOp!^dense_3820/MatMul/ReadVariableOp1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp"^dense_3821/BiasAdd/ReadVariableOp!^dense_3821/MatMul/ReadVariableOp1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp"^dense_3822/BiasAdd/ReadVariableOp!^dense_3822/MatMul/ReadVariableOp1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp"^dense_3823/BiasAdd/ReadVariableOp!^dense_3823/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_3820/BiasAdd/ReadVariableOp!dense_3820/BiasAdd/ReadVariableOp2D
 dense_3820/MatMul/ReadVariableOp dense_3820/MatMul/ReadVariableOp2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_3821/BiasAdd/ReadVariableOp!dense_3821/BiasAdd/ReadVariableOp2D
 dense_3821/MatMul/ReadVariableOp dense_3821/MatMul/ReadVariableOp2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_3822/BiasAdd/ReadVariableOp!dense_3822/BiasAdd/ReadVariableOp2D
 dense_3822/MatMul/ReadVariableOp dense_3822/MatMul/ReadVariableOp2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_3823/BiasAdd/ReadVariableOp!dense_3823/BiasAdd/ReadVariableOp2D
 dense_3823/MatMul/ReadVariableOp dense_3823/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647217

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647816

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
0__inference_sequential_928_layer_call_fn_8647583

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_928_layer_call_and_return_conditional_losses_86473642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647364

inputs$
dense_3820_8647322:  
dense_3820_8647324: $
dense_3821_8647328:  
dense_3821_8647330:$
dense_3822_8647334: 
dense_3822_8647336:$
dense_3823_8647340: 
dense_3823_8647342:
identity??"dense_3820/StatefulPartitionedCall?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?"dense_3821/StatefulPartitionedCall?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?"dense_3822/StatefulPartitionedCall?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?"dense_3823/StatefulPartitionedCall?$dropout_2895/StatefulPartitionedCall?$dropout_2896/StatefulPartitionedCall?$dropout_2897/StatefulPartitionedCall?
"dense_3820/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3820_8647322dense_3820_8647324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3820_layer_call_and_return_conditional_losses_86470592$
"dense_3820/StatefulPartitionedCall?
$dropout_2895/StatefulPartitionedCallStatefulPartitionedCall+dense_3820/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2895_layer_call_and_return_conditional_losses_86472832&
$dropout_2895/StatefulPartitionedCall?
"dense_3821/StatefulPartitionedCallStatefulPartitionedCall-dropout_2895/StatefulPartitionedCall:output:0dense_3821_8647328dense_3821_8647330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3821_layer_call_and_return_conditional_losses_86470892$
"dense_3821/StatefulPartitionedCall?
$dropout_2896/StatefulPartitionedCallStatefulPartitionedCall+dense_3821/StatefulPartitionedCall:output:0%^dropout_2895/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2896_layer_call_and_return_conditional_losses_86472502&
$dropout_2896/StatefulPartitionedCall?
"dense_3822/StatefulPartitionedCallStatefulPartitionedCall-dropout_2896/StatefulPartitionedCall:output:0dense_3822_8647334dense_3822_8647336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3822_layer_call_and_return_conditional_losses_86471192$
"dense_3822/StatefulPartitionedCall?
$dropout_2897/StatefulPartitionedCallStatefulPartitionedCall+dense_3822/StatefulPartitionedCall:output:0%^dropout_2896/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2897_layer_call_and_return_conditional_losses_86472172&
$dropout_2897/StatefulPartitionedCall?
"dense_3823/StatefulPartitionedCallStatefulPartitionedCall-dropout_2897/StatefulPartitionedCall:output:0dense_3823_8647340dense_3823_8647342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3823_layer_call_and_return_conditional_losses_86471432$
"dense_3823/StatefulPartitionedCall?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3820_8647322*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mul?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3821_8647328*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mul?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3822_8647334*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mul?
IdentityIdentity+dense_3823/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^dense_3820/StatefulPartitionedCall1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp#^dense_3821/StatefulPartitionedCall1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp#^dense_3822/StatefulPartitionedCall1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp#^dense_3823/StatefulPartitionedCall%^dropout_2895/StatefulPartitionedCall%^dropout_2896/StatefulPartitionedCall%^dropout_2897/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_3820/StatefulPartitionedCall"dense_3820/StatefulPartitionedCall2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3821/StatefulPartitionedCall"dense_3821/StatefulPartitionedCall2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3822/StatefulPartitionedCall"dense_3822/StatefulPartitionedCall2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3823/StatefulPartitionedCall"dense_3823/StatefulPartitionedCall2L
$dropout_2895/StatefulPartitionedCall$dropout_2895/StatefulPartitionedCall2L
$dropout_2896/StatefulPartitionedCall$dropout_2896/StatefulPartitionedCall2L
$dropout_2897/StatefulPartitionedCall$dropout_2897/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647250

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647494	
input$
dense_3820_8647452:  
dense_3820_8647454: $
dense_3821_8647458:  
dense_3821_8647460:$
dense_3822_8647464: 
dense_3822_8647466:$
dense_3823_8647470: 
dense_3823_8647472:
identity??"dense_3820/StatefulPartitionedCall?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?"dense_3821/StatefulPartitionedCall?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?"dense_3822/StatefulPartitionedCall?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?"dense_3823/StatefulPartitionedCall?$dropout_2895/StatefulPartitionedCall?$dropout_2896/StatefulPartitionedCall?$dropout_2897/StatefulPartitionedCall?
"dense_3820/StatefulPartitionedCallStatefulPartitionedCallinputdense_3820_8647452dense_3820_8647454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3820_layer_call_and_return_conditional_losses_86470592$
"dense_3820/StatefulPartitionedCall?
$dropout_2895/StatefulPartitionedCallStatefulPartitionedCall+dense_3820/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2895_layer_call_and_return_conditional_losses_86472832&
$dropout_2895/StatefulPartitionedCall?
"dense_3821/StatefulPartitionedCallStatefulPartitionedCall-dropout_2895/StatefulPartitionedCall:output:0dense_3821_8647458dense_3821_8647460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3821_layer_call_and_return_conditional_losses_86470892$
"dense_3821/StatefulPartitionedCall?
$dropout_2896/StatefulPartitionedCallStatefulPartitionedCall+dense_3821/StatefulPartitionedCall:output:0%^dropout_2895/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2896_layer_call_and_return_conditional_losses_86472502&
$dropout_2896/StatefulPartitionedCall?
"dense_3822/StatefulPartitionedCallStatefulPartitionedCall-dropout_2896/StatefulPartitionedCall:output:0dense_3822_8647464dense_3822_8647466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3822_layer_call_and_return_conditional_losses_86471192$
"dense_3822/StatefulPartitionedCall?
$dropout_2897/StatefulPartitionedCallStatefulPartitionedCall+dense_3822/StatefulPartitionedCall:output:0%^dropout_2896/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2897_layer_call_and_return_conditional_losses_86472172&
$dropout_2897/StatefulPartitionedCall?
"dense_3823/StatefulPartitionedCallStatefulPartitionedCall-dropout_2897/StatefulPartitionedCall:output:0dense_3823_8647470dense_3823_8647472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3823_layer_call_and_return_conditional_losses_86471432$
"dense_3823/StatefulPartitionedCall?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3820_8647452*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mul?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3821_8647458*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mul?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3822_8647464*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mul?
IdentityIdentity+dense_3823/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^dense_3820/StatefulPartitionedCall1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp#^dense_3821/StatefulPartitionedCall1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp#^dense_3822/StatefulPartitionedCall1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp#^dense_3823/StatefulPartitionedCall%^dropout_2895/StatefulPartitionedCall%^dropout_2896/StatefulPartitionedCall%^dropout_2897/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_3820/StatefulPartitionedCall"dense_3820/StatefulPartitionedCall2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3821/StatefulPartitionedCall"dense_3821/StatefulPartitionedCall2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3822/StatefulPartitionedCall"dense_3822/StatefulPartitionedCall2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3823/StatefulPartitionedCall"dense_3823/StatefulPartitionedCall2L
$dropout_2895/StatefulPartitionedCall$dropout_2895/StatefulPartitionedCall2L
$dropout_2896/StatefulPartitionedCall$dropout_2896/StatefulPartitionedCall2L
$dropout_2897/StatefulPartitionedCall$dropout_2897/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
h
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647283

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_dense_3823_layer_call_and_return_conditional_losses_8647143

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647769

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_8648171
file_prefix4
"assignvariableop_dense_3820_kernel: 0
"assignvariableop_1_dense_3820_bias: 6
$assignvariableop_2_dense_3821_kernel: 0
"assignvariableop_3_dense_3821_bias:6
$assignvariableop_4_dense_3822_kernel:0
"assignvariableop_5_dense_3822_bias:6
$assignvariableop_6_dense_3823_kernel:0
"assignvariableop_7_dense_3823_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: >
,assignvariableop_17_adam_dense_3820_kernel_m: 8
*assignvariableop_18_adam_dense_3820_bias_m: >
,assignvariableop_19_adam_dense_3821_kernel_m: 8
*assignvariableop_20_adam_dense_3821_bias_m:>
,assignvariableop_21_adam_dense_3822_kernel_m:8
*assignvariableop_22_adam_dense_3822_bias_m:>
,assignvariableop_23_adam_dense_3823_kernel_m:8
*assignvariableop_24_adam_dense_3823_bias_m:>
,assignvariableop_25_adam_dense_3820_kernel_v: 8
*assignvariableop_26_adam_dense_3820_bias_v: >
,assignvariableop_27_adam_dense_3821_kernel_v: 8
*assignvariableop_28_adam_dense_3821_bias_v:>
,assignvariableop_29_adam_dense_3822_kernel_v:8
*assignvariableop_30_adam_dense_3822_bias_v:>
,assignvariableop_31_adam_dense_3823_kernel_v:8
*assignvariableop_32_adam_dense_3823_bias_v:
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_dense_3820_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_3820_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_3821_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_3821_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_3822_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_3822_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense_3823_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_3823_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_dense_3820_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_3820_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense_3821_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_3821_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_3822_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_3822_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_dense_3823_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_3823_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_dense_3820_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_3820_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_dense_3821_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_3821_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_dense_3822_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_3822_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_dense_3823_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_3823_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33f
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_34?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
G__inference_dense_3821_layer_call_and_return_conditional_losses_8647089

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
g
.__inference_dropout_2897_layer_call_fn_8647870

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2897_layer_call_and_return_conditional_losses_86472172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647757

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
J
.__inference_dropout_2895_layer_call_fn_8647747

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2895_layer_call_and_return_conditional_losses_86470702
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
0__inference_sequential_928_layer_call_fn_8647562

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_928_layer_call_and_return_conditional_losses_86471682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_3823_layer_call_and_return_conditional_losses_8647907

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?I
?
 __inference__traced_save_8648062
file_prefix0
,savev2_dense_3820_kernel_read_readvariableop.
*savev2_dense_3820_bias_read_readvariableop0
,savev2_dense_3821_kernel_read_readvariableop.
*savev2_dense_3821_bias_read_readvariableop0
,savev2_dense_3822_kernel_read_readvariableop.
*savev2_dense_3822_bias_read_readvariableop0
,savev2_dense_3823_kernel_read_readvariableop.
*savev2_dense_3823_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_dense_3820_kernel_m_read_readvariableop5
1savev2_adam_dense_3820_bias_m_read_readvariableop7
3savev2_adam_dense_3821_kernel_m_read_readvariableop5
1savev2_adam_dense_3821_bias_m_read_readvariableop7
3savev2_adam_dense_3822_kernel_m_read_readvariableop5
1savev2_adam_dense_3822_bias_m_read_readvariableop7
3savev2_adam_dense_3823_kernel_m_read_readvariableop5
1savev2_adam_dense_3823_bias_m_read_readvariableop7
3savev2_adam_dense_3820_kernel_v_read_readvariableop5
1savev2_adam_dense_3820_bias_v_read_readvariableop7
3savev2_adam_dense_3821_kernel_v_read_readvariableop5
1savev2_adam_dense_3821_bias_v_read_readvariableop7
3savev2_adam_dense_3822_kernel_v_read_readvariableop5
1savev2_adam_dense_3822_bias_v_read_readvariableop7
3savev2_adam_dense_3823_kernel_v_read_readvariableop5
1savev2_adam_dense_3823_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_3820_kernel_read_readvariableop*savev2_dense_3820_bias_read_readvariableop,savev2_dense_3821_kernel_read_readvariableop*savev2_dense_3821_bias_read_readvariableop,savev2_dense_3822_kernel_read_readvariableop*savev2_dense_3822_bias_read_readvariableop,savev2_dense_3823_kernel_read_readvariableop*savev2_dense_3823_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_dense_3820_kernel_m_read_readvariableop1savev2_adam_dense_3820_bias_m_read_readvariableop3savev2_adam_dense_3821_kernel_m_read_readvariableop1savev2_adam_dense_3821_bias_m_read_readvariableop3savev2_adam_dense_3822_kernel_m_read_readvariableop1savev2_adam_dense_3822_bias_m_read_readvariableop3savev2_adam_dense_3823_kernel_m_read_readvariableop1savev2_adam_dense_3823_bias_m_read_readvariableop3savev2_adam_dense_3820_kernel_v_read_readvariableop1savev2_adam_dense_3820_bias_v_read_readvariableop3savev2_adam_dense_3821_kernel_v_read_readvariableop1savev2_adam_dense_3821_bias_v_read_readvariableop3savev2_adam_dense_3822_kernel_v_read_readvariableop1savev2_adam_dense_3822_bias_v_read_readvariableop3savev2_adam_dense_3823_kernel_v_read_readvariableop1savev2_adam_dense_3823_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
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

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : :::::: : : : : : : : : : : : :::::: : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::"

_output_shapes
: 
?i
?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647710

inputs;
)dense_3820_matmul_readvariableop_resource: 8
*dense_3820_biasadd_readvariableop_resource: ;
)dense_3821_matmul_readvariableop_resource: 8
*dense_3821_biasadd_readvariableop_resource:;
)dense_3822_matmul_readvariableop_resource:8
*dense_3822_biasadd_readvariableop_resource:;
)dense_3823_matmul_readvariableop_resource:8
*dense_3823_biasadd_readvariableop_resource:
identity??!dense_3820/BiasAdd/ReadVariableOp? dense_3820/MatMul/ReadVariableOp?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?!dense_3821/BiasAdd/ReadVariableOp? dense_3821/MatMul/ReadVariableOp?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?!dense_3822/BiasAdd/ReadVariableOp? dense_3822/MatMul/ReadVariableOp?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?!dense_3823/BiasAdd/ReadVariableOp? dense_3823/MatMul/ReadVariableOp?
 dense_3820/MatMul/ReadVariableOpReadVariableOp)dense_3820_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_3820/MatMul/ReadVariableOp?
dense_3820/MatMulMatMulinputs(dense_3820/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_3820/MatMul?
!dense_3820/BiasAdd/ReadVariableOpReadVariableOp*dense_3820_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!dense_3820/BiasAdd/ReadVariableOp?
dense_3820/BiasAddBiasAdddense_3820/MatMul:product:0)dense_3820/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_3820/BiasAddy
dense_3820/ReluReludense_3820/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_3820/Relu}
dropout_2895/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2895/dropout/Const?
dropout_2895/dropout/MulMuldense_3820/Relu:activations:0#dropout_2895/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_2895/dropout/Mul?
dropout_2895/dropout/ShapeShapedense_3820/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2895/dropout/Shape?
1dropout_2895/dropout/random_uniform/RandomUniformRandomUniform#dropout_2895/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype023
1dropout_2895/dropout/random_uniform/RandomUniform?
#dropout_2895/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2%
#dropout_2895/dropout/GreaterEqual/y?
!dropout_2895/dropout/GreaterEqualGreaterEqual:dropout_2895/dropout/random_uniform/RandomUniform:output:0,dropout_2895/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2#
!dropout_2895/dropout/GreaterEqual?
dropout_2895/dropout/CastCast%dropout_2895/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_2895/dropout/Cast?
dropout_2895/dropout/Mul_1Muldropout_2895/dropout/Mul:z:0dropout_2895/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_2895/dropout/Mul_1?
 dense_3821/MatMul/ReadVariableOpReadVariableOp)dense_3821_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_3821/MatMul/ReadVariableOp?
dense_3821/MatMulMatMuldropout_2895/dropout/Mul_1:z:0(dense_3821/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3821/MatMul?
!dense_3821/BiasAdd/ReadVariableOpReadVariableOp*dense_3821_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3821/BiasAdd/ReadVariableOp?
dense_3821/BiasAddBiasAdddense_3821/MatMul:product:0)dense_3821/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3821/BiasAddy
dense_3821/ReluReludense_3821/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3821/Relu}
dropout_2896/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2896/dropout/Const?
dropout_2896/dropout/MulMuldense_3821/Relu:activations:0#dropout_2896/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_2896/dropout/Mul?
dropout_2896/dropout/ShapeShapedense_3821/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2896/dropout/Shape?
1dropout_2896/dropout/random_uniform/RandomUniformRandomUniform#dropout_2896/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype023
1dropout_2896/dropout/random_uniform/RandomUniform?
#dropout_2896/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2%
#dropout_2896/dropout/GreaterEqual/y?
!dropout_2896/dropout/GreaterEqualGreaterEqual:dropout_2896/dropout/random_uniform/RandomUniform:output:0,dropout_2896/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2#
!dropout_2896/dropout/GreaterEqual?
dropout_2896/dropout/CastCast%dropout_2896/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_2896/dropout/Cast?
dropout_2896/dropout/Mul_1Muldropout_2896/dropout/Mul:z:0dropout_2896/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_2896/dropout/Mul_1?
 dense_3822/MatMul/ReadVariableOpReadVariableOp)dense_3822_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3822/MatMul/ReadVariableOp?
dense_3822/MatMulMatMuldropout_2896/dropout/Mul_1:z:0(dense_3822/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3822/MatMul?
!dense_3822/BiasAdd/ReadVariableOpReadVariableOp*dense_3822_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3822/BiasAdd/ReadVariableOp?
dense_3822/BiasAddBiasAdddense_3822/MatMul:product:0)dense_3822/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3822/BiasAddy
dense_3822/ReluReludense_3822/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3822/Relu}
dropout_2897/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2897/dropout/Const?
dropout_2897/dropout/MulMuldense_3822/Relu:activations:0#dropout_2897/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_2897/dropout/Mul?
dropout_2897/dropout/ShapeShapedense_3822/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2897/dropout/Shape?
1dropout_2897/dropout/random_uniform/RandomUniformRandomUniform#dropout_2897/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype023
1dropout_2897/dropout/random_uniform/RandomUniform?
#dropout_2897/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2%
#dropout_2897/dropout/GreaterEqual/y?
!dropout_2897/dropout/GreaterEqualGreaterEqual:dropout_2897/dropout/random_uniform/RandomUniform:output:0,dropout_2897/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2#
!dropout_2897/dropout/GreaterEqual?
dropout_2897/dropout/CastCast%dropout_2897/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_2897/dropout/Cast?
dropout_2897/dropout/Mul_1Muldropout_2897/dropout/Mul:z:0dropout_2897/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_2897/dropout/Mul_1?
 dense_3823/MatMul/ReadVariableOpReadVariableOp)dense_3823_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3823/MatMul/ReadVariableOp?
dense_3823/MatMulMatMuldropout_2897/dropout/Mul_1:z:0(dense_3823/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3823/MatMul?
!dense_3823/BiasAdd/ReadVariableOpReadVariableOp*dense_3823_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3823/BiasAdd/ReadVariableOp?
dense_3823/BiasAddBiasAdddense_3823/MatMul:product:0)dense_3823/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3823/BiasAdd?
dense_3823/SoftmaxSoftmaxdense_3823/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3823/Softmax?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp)dense_3820_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mul?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp)dense_3821_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mul?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp)dense_3822_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mulw
IdentityIdentitydense_3823/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_3820/BiasAdd/ReadVariableOp!^dense_3820/MatMul/ReadVariableOp1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp"^dense_3821/BiasAdd/ReadVariableOp!^dense_3821/MatMul/ReadVariableOp1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp"^dense_3822/BiasAdd/ReadVariableOp!^dense_3822/MatMul/ReadVariableOp1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp"^dense_3823/BiasAdd/ReadVariableOp!^dense_3823/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2F
!dense_3820/BiasAdd/ReadVariableOp!dense_3820/BiasAdd/ReadVariableOp2D
 dense_3820/MatMul/ReadVariableOp dense_3820/MatMul/ReadVariableOp2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_3821/BiasAdd/ReadVariableOp!dense_3821/BiasAdd/ReadVariableOp2D
 dense_3821/MatMul/ReadVariableOp dense_3821/MatMul/ReadVariableOp2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_3822/BiasAdd/ReadVariableOp!dense_3822/BiasAdd/ReadVariableOp2D
 dense_3822/MatMul/ReadVariableOp dense_3822/MatMul/ReadVariableOp2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_3823/BiasAdd/ReadVariableOp!dense_3823/BiasAdd/ReadVariableOp2D
 dense_3823/MatMul/ReadVariableOp dense_3823/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647100

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_3820_layer_call_fn_8647725

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3820_layer_call_and_return_conditional_losses_86470592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647887

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_dropout_2896_layer_call_fn_8647806

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2896_layer_call_and_return_conditional_losses_86471002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647168

inputs$
dense_3820_8647060:  
dense_3820_8647062: $
dense_3821_8647090:  
dense_3821_8647092:$
dense_3822_8647120: 
dense_3822_8647122:$
dense_3823_8647144: 
dense_3823_8647146:
identity??"dense_3820/StatefulPartitionedCall?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?"dense_3821/StatefulPartitionedCall?0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?"dense_3822/StatefulPartitionedCall?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?"dense_3823/StatefulPartitionedCall?
"dense_3820/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3820_8647060dense_3820_8647062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3820_layer_call_and_return_conditional_losses_86470592$
"dense_3820/StatefulPartitionedCall?
dropout_2895/PartitionedCallPartitionedCall+dense_3820/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2895_layer_call_and_return_conditional_losses_86470702
dropout_2895/PartitionedCall?
"dense_3821/StatefulPartitionedCallStatefulPartitionedCall%dropout_2895/PartitionedCall:output:0dense_3821_8647090dense_3821_8647092*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3821_layer_call_and_return_conditional_losses_86470892$
"dense_3821/StatefulPartitionedCall?
dropout_2896/PartitionedCallPartitionedCall+dense_3821/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2896_layer_call_and_return_conditional_losses_86471002
dropout_2896/PartitionedCall?
"dense_3822/StatefulPartitionedCallStatefulPartitionedCall%dropout_2896/PartitionedCall:output:0dense_3822_8647120dense_3822_8647122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3822_layer_call_and_return_conditional_losses_86471192$
"dense_3822/StatefulPartitionedCall?
dropout_2897/PartitionedCallPartitionedCall+dense_3822/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2897_layer_call_and_return_conditional_losses_86471302
dropout_2897/PartitionedCall?
"dense_3823/StatefulPartitionedCallStatefulPartitionedCall%dropout_2897/PartitionedCall:output:0dense_3823_8647144dense_3823_8647146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3823_layer_call_and_return_conditional_losses_86471432$
"dense_3823/StatefulPartitionedCall?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3820_8647060*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mul?
0dense_3821/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3821_8647090*
_output_shapes

: *
dtype022
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3821/kernel/Regularizer/AbsAbs8dense_3821/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3821/kernel/Regularizer/Abs?
#dense_3821/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3821/kernel/Regularizer/Const?
!dense_3821/kernel/Regularizer/SumSum%dense_3821/kernel/Regularizer/Abs:y:0,dense_3821/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/Sum?
#dense_3821/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3821/kernel/Regularizer/mul/x?
!dense_3821/kernel/Regularizer/mulMul,dense_3821/kernel/Regularizer/mul/x:output:0*dense_3821/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3821/kernel/Regularizer/mul?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_3822_8647120*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mul?
IdentityIdentity+dense_3823/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp#^dense_3820/StatefulPartitionedCall1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp#^dense_3821/StatefulPartitionedCall1^dense_3821/kernel/Regularizer/Abs/ReadVariableOp#^dense_3822/StatefulPartitionedCall1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp#^dense_3823/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2H
"dense_3820/StatefulPartitionedCall"dense_3820/StatefulPartitionedCall2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3821/StatefulPartitionedCall"dense_3821/StatefulPartitionedCall2d
0dense_3821/kernel/Regularizer/Abs/ReadVariableOp0dense_3821/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3822/StatefulPartitionedCall"dense_3822/StatefulPartitionedCall2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp2H
"dense_3823/StatefulPartitionedCall"dense_3823/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647130

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_3822_layer_call_and_return_conditional_losses_8647860

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_3820_layer_call_and_return_conditional_losses_8647059

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_3822_layer_call_and_return_conditional_losses_8647119

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
0dense_3822/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3822/kernel/Regularizer/AbsAbs8dense_3822/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:2#
!dense_3822/kernel/Regularizer/Abs?
#dense_3822/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3822/kernel/Regularizer/Const?
!dense_3822/kernel/Regularizer/SumSum%dense_3822/kernel/Regularizer/Abs:y:0,dense_3822/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/Sum?
#dense_3822/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3822/kernel/Regularizer/mul/x?
!dense_3822/kernel/Regularizer/mulMul,dense_3822/kernel/Regularizer/mul/x:output:0*dense_3822/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3822/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3822/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3822/kernel/Regularizer/Abs/ReadVariableOp0dense_3822/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
0__inference_sequential_928_layer_call_fn_8647404	
input
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_sequential_928_layer_call_and_return_conditional_losses_86473642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
g
.__inference_dropout_2896_layer_call_fn_8647811

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dropout_2896_layer_call_and_return_conditional_losses_86472502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_3821_layer_call_fn_8647784

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_3821_layer_call_and_return_conditional_losses_86470892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_dense_3820_layer_call_and_return_conditional_losses_8647742

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
0dense_3820/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp?
!dense_3820/kernel/Regularizer/AbsAbs8dense_3820/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_3820/kernel/Regularizer/Abs?
#dense_3820/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_3820/kernel/Regularizer/Const?
!dense_3820/kernel/Regularizer/SumSum%dense_3820/kernel/Regularizer/Abs:y:0,dense_3820/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/Sum?
#dense_3820/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82%
#dense_3820/kernel/Regularizer/mul/x?
!dense_3820/kernel/Regularizer/mulMul,dense_3820/kernel/Regularizer/mul/x:output:0*dense_3820/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_3820/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3820/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3820/kernel/Regularizer/Abs/ReadVariableOp0dense_3820/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input.
serving_default_input:0?????????>

dense_38230
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
z_default_save_signature
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_sequential
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
regularization_losses
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(	variables
)regularization_losses
*trainable_variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm"mn#mo,mp-mqvrvsvtvu"vv#vw,vx-vy"
	optimizer
X
0
1
2
3
"4
#5
,6
-7"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
X
0
1
2
3
"4
#5
,6
-7"
trackable_list_wrapper
?
7layer_metrics
		variables
8metrics

regularization_losses
trainable_variables
9non_trainable_variables
:layer_regularization_losses

;layers
{__call__
z_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:! 2dense_3820/kernel
: 2dense_3820/bias
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
<layer_metrics
	variables
=metrics
regularization_losses
trainable_variables
>non_trainable_variables
?layer_regularization_losses

@layers
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Alayer_metrics
	variables
Bmetrics
regularization_losses
trainable_variables
Cnon_trainable_variables
Dlayer_regularization_losses

Elayers
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:! 2dense_3821/kernel
:2dense_3821/bias
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Flayer_metrics
	variables
Gmetrics
regularization_losses
trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses

Jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Klayer_metrics
	variables
Lmetrics
regularization_losses
 trainable_variables
Mnon_trainable_variables
Nlayer_regularization_losses

Olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!2dense_3822/kernel
:2dense_3822/bias
.
"0
#1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
Player_metrics
$	variables
Qmetrics
%regularization_losses
&trainable_variables
Rnon_trainable_variables
Slayer_regularization_losses

Tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ulayer_metrics
(	variables
Vmetrics
)regularization_losses
*trainable_variables
Wnon_trainable_variables
Xlayer_regularization_losses

Ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!2dense_3823/kernel
:2dense_3823/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
Zlayer_metrics
.	variables
[metrics
/regularization_losses
0trainable_variables
\non_trainable_variables
]layer_regularization_losses

^layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
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
(
?0"
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
(
?0"
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
N
	atotal
	bcount
c	variables
d	keras_api"
_tf_keras_metric
^
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
(:& 2Adam/dense_3820/kernel/m
":  2Adam/dense_3820/bias/m
(:& 2Adam/dense_3821/kernel/m
": 2Adam/dense_3821/bias/m
(:&2Adam/dense_3822/kernel/m
": 2Adam/dense_3822/bias/m
(:&2Adam/dense_3823/kernel/m
": 2Adam/dense_3823/bias/m
(:& 2Adam/dense_3820/kernel/v
":  2Adam/dense_3820/bias/v
(:& 2Adam/dense_3821/kernel/v
": 2Adam/dense_3821/bias/v
(:&2Adam/dense_3822/kernel/v
": 2Adam/dense_3822/bias/v
(:&2Adam/dense_3823/kernel/v
": 2Adam/dense_3823/bias/v
?B?
"__inference__wrapped_model_8647035input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_sequential_928_layer_call_fn_8647187
0__inference_sequential_928_layer_call_fn_8647562
0__inference_sequential_928_layer_call_fn_8647583
0__inference_sequential_928_layer_call_fn_8647404?
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
?2?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647636
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647710
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647449
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647494?
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
,__inference_dense_3820_layer_call_fn_8647725?
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
G__inference_dense_3820_layer_call_and_return_conditional_losses_8647742?
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
.__inference_dropout_2895_layer_call_fn_8647747
.__inference_dropout_2895_layer_call_fn_8647752?
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
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647757
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647769?
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
,__inference_dense_3821_layer_call_fn_8647784?
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
G__inference_dense_3821_layer_call_and_return_conditional_losses_8647801?
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
.__inference_dropout_2896_layer_call_fn_8647806
.__inference_dropout_2896_layer_call_fn_8647811?
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
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647816
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647828?
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
,__inference_dense_3822_layer_call_fn_8647843?
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
G__inference_dense_3822_layer_call_and_return_conditional_losses_8647860?
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
.__inference_dropout_2897_layer_call_fn_8647865
.__inference_dropout_2897_layer_call_fn_8647870?
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
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647875
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647887?
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
,__inference_dense_3823_layer_call_fn_8647896?
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
G__inference_dense_3823_layer_call_and_return_conditional_losses_8647907?
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
__inference_loss_fn_0_8647918?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_8647929?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_8647940?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
%__inference_signature_wrapper_8647541input"?
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
"__inference__wrapped_model_8647035s"#,-.?+
$?!
?
input?????????
? "7?4
2

dense_3823$?!

dense_3823??????????
G__inference_dense_3820_layer_call_and_return_conditional_losses_8647742\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? 
,__inference_dense_3820_layer_call_fn_8647725O/?,
%?"
 ?
inputs?????????
? "?????????? ?
G__inference_dense_3821_layer_call_and_return_conditional_losses_8647801\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
,__inference_dense_3821_layer_call_fn_8647784O/?,
%?"
 ?
inputs????????? 
? "???????????
G__inference_dense_3822_layer_call_and_return_conditional_losses_8647860\"#/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_3822_layer_call_fn_8647843O"#/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_3823_layer_call_and_return_conditional_losses_8647907\,-/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_3823_layer_call_fn_8647896O,-/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647757\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
I__inference_dropout_2895_layer_call_and_return_conditional_losses_8647769\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
.__inference_dropout_2895_layer_call_fn_8647747O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
.__inference_dropout_2895_layer_call_fn_8647752O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647816\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
I__inference_dropout_2896_layer_call_and_return_conditional_losses_8647828\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
.__inference_dropout_2896_layer_call_fn_8647806O3?0
)?&
 ?
inputs?????????
p 
? "???????????
.__inference_dropout_2896_layer_call_fn_8647811O3?0
)?&
 ?
inputs?????????
p
? "???????????
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647875\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
I__inference_dropout_2897_layer_call_and_return_conditional_losses_8647887\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
.__inference_dropout_2897_layer_call_fn_8647865O3?0
)?&
 ?
inputs?????????
p 
? "???????????
.__inference_dropout_2897_layer_call_fn_8647870O3?0
)?&
 ?
inputs?????????
p
? "??????????<
__inference_loss_fn_0_8647918?

? 
? "? <
__inference_loss_fn_1_8647929?

? 
? "? <
__inference_loss_fn_2_8647940"?

? 
? "? ?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647449i"#,-6?3
,?)
?
input?????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647494i"#,-6?3
,?)
?
input?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647636j"#,-7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_sequential_928_layer_call_and_return_conditional_losses_8647710j"#,-7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
0__inference_sequential_928_layer_call_fn_8647187\"#,-6?3
,?)
?
input?????????
p 

 
? "???????????
0__inference_sequential_928_layer_call_fn_8647404\"#,-6?3
,?)
?
input?????????
p

 
? "???????????
0__inference_sequential_928_layer_call_fn_8647562]"#,-7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
0__inference_sequential_928_layer_call_fn_8647583]"#,-7?4
-?*
 ?
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_8647541|"#,-7?4
? 
-?*
(
input?
input?????????"7?4
2

dense_3823$?!

dense_3823?????????
import logging
import numpy as np
import tensorflow as tf

from ..utils.op_registry import OPERATOR
from . import dimension_utils
import keras

LOG = logging.getLogger("calculations_layers :")

def np2tf(x):
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x, False
    return x, True

def match_tensor(x1:tf.Tensor or np.ndarray, x2:tf.Tensor or np.ndarray):

    x1, f1 = np2tf(x1)
    x2, f2 = np2tf(x2)

    # no need to transpose if all var are tensor, we assume tensor are computed by gragh.
    if f1 and f2:
        return x1, x2

    # ensure tensor is set to x1, weights set to x2
    if f2:
        x1, x2 = x2, x1

    # if x1.shape.ndims != x2.shape.ndims:
    #     while x2.shape.ndims < x1.shape.ndims:
    #         x2 = tf.expand_dims(x2, axis=0)
    if len(x1.shape) != len(x2.shape):
        while len(x2.shape) < len(x1.shape):
            x2 = tf.expand_dims(x2, axis=0)
    
    # new_shape = dimension_utils.shape_NCD_to_NDC_format([i for i in range(len(x2.shape))])
    # x2 = tf.transpose(x2, new_shape)
    return (x2, x1) if f2 else (x1, x2)


@OPERATOR.register_operator("Add")
class TFAdd(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def call(self, first_operand, second_operand,*args, **kwargs):
        return keras.ops.add(first_operand, second_operand)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
        })
        return config

@OPERATOR.register_operator("Sub")
class TFSub(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def call(self, first_operand, second_operand,*args, **kwargs):
        return keras.ops.subtract(first_operand, second_operand)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
        })
        return config

@OPERATOR.register_operator("Mul")
class TFMul(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def call(self, first_operand, second_operand,*args, **kwargs):
        return keras.ops.multiply(first_operand, second_operand)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
        })
        return config

@OPERATOR.register_operator("Div")
class TFDiv(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self,first_operand, second_operand, *args, **kwargs):
        return keras.ops.divide(first_operand, second_operand)


    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
        })
        return config
    
@OPERATOR.register_operator("Equal")
class TFEqual(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def call(self, first_operand, second_operand,*args, **kwargs):
        return keras.ops.equal(first_operand, second_operand)

    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
        })
        return config

@OPERATOR.register_operator("Less")
class TFLess(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self,first_operand, second_operand, *args, **kwargs):
        return keras.ops.less(first_operand, second_operand)

    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
        })
        return config
    
@OPERATOR.register_operator("Greater")
class TFGreater(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self,first_operand, second_operand, *args, **kwargs):
        return keras.ops.greater(first_operand, second_operand)
    
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,

        })
        return config

    
@OPERATOR.register_operator("Where")
class TFWhere(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.true_value = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.false_value = tensor_grap[node_inputs[2]] if node_inputs[2] in tensor_grap else node_weights[node_inputs[2]]
        self.true_value, self.false_value = match_tensor(self.true_value, self.false_value)


    def call(self, condition, true_value, false_value, *args,**kwargs):
        return keras.ops.where(condition, true_value, false_value)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "true_value": self.true_value,
             "false_value": self.false_value
        })
        return config
    
@OPERATOR.register_operator("Not")
class TFNot(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()


    def call(self,input, *args, **kwargs):
        return keras.ops.logical_not(input)

@OPERATOR.register_operator("And")
class TFAnd(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self,  *args, **kwargs):
        return keras.ops.logical_and(args[0], args[1])

@OPERATOR.register_operator("Or")
class TFOr(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self,  *args, **kwargs):
        return keras.ops.logical_or(args[0], args[1])


@OPERATOR.register_operator("Abs")
class TFAbs(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self,input,  *args, **kwargs):
        return keras.ops.absolute(input)

@OPERATOR.register_operator("Neg")
class TFNeg(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self,input,  *args, **kwargs):
        return 0-input

@OPERATOR.register_operator("Reciprocal")
class TFReciprocal(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return keras.ops.reciprocal(inputs)

@OPERATOR.register_operator("Sqrt")
class TFSqrt(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return keras.ops.sqrt(inputs)

@OPERATOR.register_operator("Exp")
class TFExp(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def cal(self, inputs, *args, **kwargs):
        return keras.ops.exp(inputs)


@OPERATOR.register_operator("Log")
class TFLog(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return keras.ops.log(inputs)

@OPERATOR.register_operator("Floor")
class TFFloor(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return keras.ops.floor(inputs)
    
@OPERATOR.register_operator("Ceil")
class TFCeil(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return keras.ops.ceil(inputs)


@OPERATOR.register_operator("ReduceMax")
class TFReduceMax(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        self.axes = node_attribute.get("axes", None)
        self.initial = node_attribute.get("initial", None)

    def call(self, inputs, *args, **kwargs):
        return keras.ops.max(inputs, axis=self.axes, keepdims=self.keep_dims, initial= self.initial)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config

@OPERATOR.register_operator("ReduceMin")
class TFReduceMin(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        self.axes = node_attribute.get("axes", None)
        self.initial = node_attribute.get("initial", None)

    def call(self, inputs, *args, **kwargs):
        return keras.ops.min(inputs, axis=self.axes, keepdims=self.keep_dims, initial = self.initial)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config


@OPERATOR.register_operator("ReduceSum")
class TFReduceSum(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        self.axes = node_attribute.get("axes", None)
       
    def call(self, inputs, *args, **kwargs):
        return keras.ops.sum(inputs, axis = self.axes, keepdims=self.keep_dims)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config


@OPERATOR.register_operator("ReduceMean")
class TFReduceMean(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        self.axes = node_attribute.get("axes", None)
       
    def call(self, inputs, *args, **kwargs):
        return keras.ops.mean(inputs, axis = self.axes, keepdims=self.keep_dims)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config


@OPERATOR.register_operator("Shape")
class TFShape(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return keras.ops.array([*keras.ops.shape(inputs)])


@OPERATOR.register_operator("ConstantOfShape")
class TFConstantOfShape(keras.layers.Layer):
    def __init__(self,tensor_grap,node_weights, node_inputs,node_attribute,*args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.value = self.node_attribute['value']
        
    def call(self, inputs,*args, **kwargs):
        # print("should be one:::: ", self.node_attribute['value'][0])
        # print('type : ', type(self.node_attribute['value'][0]))
        # print('hey: ', self.value)
        # print('typppp: ', type(self.value))
        if 'config' in self.value:
            # print("configggg")
            fill_in = self.value['config']['value'][0]
        else:
            # print("numpy float")
            fill_in = self.value[0]

        # hey:  {'class_name': '__numpy__', 'config': {'value': [1.0], 'dtype': 'float32'}}
        print('inpuuutt size: ', inputs)

        print('shapeyy const size: ', keras.ops.full(inputs.shape, fill_in).shape)
        return keras.ops.full(inputs, fill_in)

    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute, 
            'value': self.value
        })
        return config



@OPERATOR.register_operator("MatMul")
class TFMatMul(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs

        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]

    def call(self,first_operand, second_operand, *args, **kwargs):
        return keras.ops.matmul(first_operand, second_operand)

    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand
        })
        return config
    
    
# TO SUPPORT LATER

# @OPERATOR.register_operator("Pow")
# class TFPow(keras.layers.Layer):
#     def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
#         super().__init__()
#         self.tensor_grap = tensor_grap
#         self.node_weights = node_weights
#         self.node_inputs = node_inputs
#         self.power_index = node_weights[node_inputs[1]]

#     def call(self, inputs, *args, **kwargs):
#         return keras.ops.power(inputs, self.power_index)
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "tensor_grap":self.tensor_grap,
#             'node_weights':self.node_weights,
#             'node_inputs':self.node_inputs
#         })
#         return config



# @OPERATOR.register_operator("ArgMax")
# class TFArgMax():
#     def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
#         super().__init__()
#         self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
#         self.keepdims = node_attribute.get("keepdims", 1) == 1

#     def __call__(self, inputs, *args, **kwargs):
#         _inputs = tf.argmax(inputs, axis=self.axis)
#         if self.keepdims:
#             _inputs = tf.expand_dims(_inputs, axis=self.axis)
#         return _inputs

# @OPERATOR.register_operator("ArgMin")
# class TFArgMin():
#     def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
#         super().__init__()
#         self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
#         self.keepdims = node_attribute.get("keepdims", 1) == 1

#     def __call__(self, inputs, *args, **kwargs):
#         _inputs = tf.argmax(inputs, axis=self.axis)
#         if self.keepdims:
#             _inputs = tf.expand_dims(_inputs, axis=self.axis)
#         return _inputs

# @OPERATOR.register_operator("Erf")
# class TFErf():
#     def __init__(self, *args, **kwargs) -> None:
#         pass
    
#     def __call__(self, inputs):
#         inputs = tf.math.erf(inputs)
#         return inputs
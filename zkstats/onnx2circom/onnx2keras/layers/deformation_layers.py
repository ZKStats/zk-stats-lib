import logging
import tensorflow as tf

from ..utils.op_registry import OPERATOR
from . import dimension_utils
import keras
LOG = logging.getLogger("deformation_layers :")

@OPERATOR.register_operator("Transpose")
class TFTranspose(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.perm_list = node_attribute['perm']

        # self.trans_in, self.trans_out = None, None
        # if kwargs.get("perm_list"):
        #     self.perm_list = kwargs.get("perm_list")
        # elif len(node_attribute['perm']) > 4:
        #     self.perm_list = []
        #     for axis in node_attribute['perm']:
        #         new_axis = dimension_utils.channel_to_last_dimension(axis)
        #         if new_axis == -1:
        #             new_axis = max(node_attribute['perm'])
        #         self.perm_list.append(new_axis)
        #     self.perm_list = dimension_utils.shape_NCD_to_NDC_format(self.perm_list)
        # else:
        #     self.perm_list = [i for i in node_attribute['perm']]
        #     LOG.info("Transpose will process tensor after change back to NCHW format.")
        #     shape_len = len(tensor_grap[node_inputs[0]].shape)
        #     self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
        #     self.trans_out = [0] + [n for n in range(2, len(self.perm_list))] + [1]

    def call(self, inputs):
        # if self.trans_in and self.trans_out:
        #     inputs = keras.ops.transpose(inputs, self.trans_in)
        #     inputs = keras.ops.transpose(inputs, self.perm_list)
        #     inputs = keras.ops.transpose(inputs, self.trans_out)
        #     return inputs
        # else:
        return keras.ops.transpose(inputs, self.perm_list)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute, 
            # 'trans_in':self.trans_in,
            'perm_list':self.perm_list,
            # 'trans_out':self.trans_out
        })
        return config
@OPERATOR.register_operator("Slice")
class TFSlice():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        if len(node_inputs) == 1:
            self.starts = node_attribute['starts'][0]
            self.ends = node_attribute['ends'][0]
            self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])
            self.steps = 1
        else:
            self.starts = node_weights[node_inputs[1]][0] if node_inputs[1] in node_weights else tensor_grap[node_inputs[1]][0]
            self.axis = node_weights[node_inputs[3]][0] if node_inputs[3] in node_weights else tensor_grap[node_inputs[3]][0]
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
            self.ends = node_weights[node_inputs[2]][0] if node_inputs[2] in node_weights else tensor_grap[node_inputs[2]][0]
            self.ends = min(self.ends, tensor_grap[node_inputs[0]].shape[self.axis])
            if len(node_inputs) < 5:
                self.steps = 1
            else:
                self.steps = node_weights[node_inputs[4]][0] if node_inputs[4] in node_weights else tensor_grap[node_inputs[4]][0]

    def __call__(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        return tf.gather(inputs, indices, axis=self.axis)


@OPERATOR.register_operator("Gather")
class TFGather(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.axis = node_attribute.get('axis', 0)
        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]

    def call(self,inputs,  *args):
        return keras.ops.take(inputs, self.indices, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute, 
            'indices':self.indices,
            'axis': self.axis
        })
        return config


@OPERATOR.register_operator("Concat")
class TFConcat(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        # self._axis = dimension_utils.channel_to_last_dimension(node_attribute['axis'])
        self._axis = node_attribute['axis']
        # self._gather = [tensor_grap[x] if x in tensor_grap else dimension_utils.tensor_NCD_to_NDC_format(node_weights[x]) for x in node_inputs]
        self._gather = [tensor_grap[x] if x in tensor_grap else node_weights[x] for x in node_inputs]
        print('gatherrrrs: ', self._gather)
    def call(self, *args, **kwargs):
        print('Call gatherrrrs: ', self._gather)
        return keras.ops.concatenate((args), axis = self._axis)
        return tf.concat(self._gather, axis=self._axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute, 
            'axis':self._axis,
            'gather':self._gather
        })
        return config

@OPERATOR.register_operator("Reshape")
class TFReshape():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.out_shape = node_weights[node_inputs[1]]
        self.trans_in, self.trans_out = None, None
        LOG.info("Reshape will process tensor after change back to NCHW format.")
        shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
        self.trans_out = [0] + [n for n in range(2, len(self.out_shape))] + [1]

    def __call__(self, inputs):
        inputs = tf.transpose(inputs, perm=self.trans_in)
        inputs = tf.reshape(inputs, shape=self.out_shape)
        inputs = tf.transpose(inputs, perm=self.trans_out)
        return inputs

@OPERATOR.register_operator("Flatten")
class TFFlatten():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        num_elements = int(tensor_grap[node_inputs[0]].shape.num_elements()/tensor_grap[node_inputs[0]].shape[0])
        input_shape = tensor_grap[node_inputs[0]].shape
        self.flat = tf.keras.layers.Flatten()
        '''
            ensure memory order match, for example:
            onnx = (B, 2, 3, 4).reshape(B, -1)
            tflite = (B, 3, 4, 2).reshape(B, -1)
            we can observe that:
            onnx.shape == tflite.shape, but np.sum(onnx-tflite) != 0
            it's cause memory order of two vars is different, we must make tflite back to onnx by transpose.
            generally, this situation is general one, below is just special situation and most appear in cnn.
            onnx = (B, 512, 1, 1)
            tflite = (B, 1, 1, 512)
            or = (B, 1, 512, 1)
            these memory order are all same.
        '''
        self.perm = None
        if num_elements != max(input_shape[1:]):
            self.perm = [0, len(input_shape)-1]
            for i in range(len(input_shape)-2):
                self.perm.append(i+1)

    def __call__(self, inputs):
        if self.perm:
            inputs = tf.transpose(inputs, perm=self.perm)
        return self.flat(inputs)

@OPERATOR.register_operator("Split")
class TFSplit(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        index = kwargs.get('index', 0)
        start = 0
        for i in range(index):
            start += int(node_attribute['split'][i])
        end = start + node_attribute['split'][index]
        self.indices = keras.ops.arange(start, end, 1)
        self.axis = node_attribute.get("axis", 0)

    def call(self, inputs):
        return keras.ops.take(inputs, indices=self.indices, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "indices": self.indices,
             "axis": self.axis
        })
        return config

@OPERATOR.register_operator("Expand")
class TFExpand():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.shape = dimension_utils.shape_NCD_to_NDC_format(node_weights[node_inputs[1]])

    def __call__(self, inputs):
        for i in range(len(self.shape)):
            if int(self.shape[i]//inputs.shape[i]) > 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]//inputs.shape[i]), axis=i)
            elif self.shape[i] < inputs.shape[i] and self.shape[i] != 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]), axis=i)
        return inputs

@OPERATOR.register_operator("Unsqueeze")
class TFUnsqueeze(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.axis = node_attribute['axes'][0]
        # self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])

    def call(self, inputs):
        return keras.ops.expand_dims(inputs, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "axis": self.axis
        })
        return config
    
@OPERATOR.register_operator("Squeeze")
class TFSqueeze(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        # self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])
        self.axis = node_attribute['axes'][0]

    def call(self, inputs):
        return keras.ops.squeeze(inputs, self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute,
            "axis": self.axis
        })
        return config

@OPERATOR.register_operator("DepthToSpace")
class TFDepthToSpace():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.block_size = node_attribute.get("blocksize", 2)
        self.mode = node_attribute.get("mode", "DCR")

    def __call__(self, inputs):
        if self.mode == "DCR":
            return tf.nn.depth_to_space(inputs, self.block_size)
        elif self.mode == "CRD":
            # help want, native tensorflow is not support CRD mode, this way will generate 5 dims op.
            b, h, w, c = inputs.shape
            tmp = tf.reshape(inputs, [b, h, w, c//(self.block_size * self.block_size), self.block_size, self.block_size])
            tmp = tf.transpose(tmp, perm=[0, 1, 4, 2, 5, 3])
            tmp = tf.reshape(tmp, [b, h*self.block_size, w*self.block_size, c//(self.block_size * self.block_size)])
            return tmp
        else:
            raise KeyError(f"For DepthToSpace, mode must be [DCR, CRD], not {self.mode}")
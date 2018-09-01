import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        #x=tf.clip_by_value(x,-1,1)
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)

def ternarize(x, thresh=0.05):
    shape = x.get_shape()
    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thresh)

    with ops.name_scope('Ternaized') as name:
        w_p = tf.get_variable('Wp', initializer=1.0, dtype=tf.float32)
        w_n = tf.get_variable('Wn', initializer=1.0, dtype=tf.float32)

        mask = tf.ones(shape)
        mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
        mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
        mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

        @tf.custom_gradient
        def _sign_mask(x):
            return tf.sign(x) * mask_z, lambda dy: dy

        w = _sign_mask(x)

        w = w * mask_np

    tf.summary.scalar(w_p.op.name + '-summary', w_p)
    tf.summary.scalar(w_n.op.name + '-summary', w_n)

    # tf.summary.histogram(w.name, w)

    return w

def BinarizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='BinarizedSpatialConvolution'):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None,[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            bin_x = binarize(x)
            ################## Just a trial ##############
            tf.summary.histogram('binarized_weight', bin_w)
            tf.summary.histogram('binarized_activations', bin_x)
            ################## Just a trial ##############
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(bin_x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv2d

def BinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        # second param for default name
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)

            out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

################# Ternarized Convolution (only weights) ######################

def TernarizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='TernarizedSpatialConvolution'):

    def ttq_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, None, [x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            ttq_w = ternarize(w)
            tf.summary.histogram('ternarized_weight', ttq_w)

            out = tf.nn.conv2d(x, ttq_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return ttq_conv2d

############### Ternarized Convolution #################################

def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=False, name='SpatialConvolution'):
    def conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name,None,[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d

def Affine(nOutputPlane, bias=True, name=None, reuse=False):
    def affineLayer(x, is_training=True):
        with tf.variable_scope(name,'Affine',[x], reuse=reuse):
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            output = tf.matmul(reshaped, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return affineLayer

def BinarizedAffine(nOutputPlane, bias=True, name=None, reuse=False):
    def b_affineLayer(x, is_training=True):
        with tf.variable_scope(name,'Affine',[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = binarize(x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return b_affineLayer

def BinarizedWeightOnlyAffine(nOutputPlane, bias=True, name=None, reuse=False):
    def bwo_affineLayer(x, is_training=True):
        with tf.variable_scope(name,'Affine',[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            w = tf.clip_by_value(w,-1,1)
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return bwo_affineLayer


################# Ternarized FC (only weights) ######################

def TernarizedAffine(nOutputPlane, bias=True, name=None, reuse=False):
    def ttq_affineLayer(x, is_training=True):
        with tf.variable_scope(name,'Affine',[x], reuse=reuse):
 
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())

            ttq_w = ternarize(w)
            tf.summary.histogram('ternarized_weight', ttq_w)
            output = tf.matmul(reshaped, ttq_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return ttq_affineLayer


################# Ternarized FC (only weights) ######################

# FC layer
def Linear(nInputPlane, nOutputPlane):
    return Affine(nInputPlane, nOutputPlane, add_bias=False)

# Used to wrap batch/instance normalization and View
def wrapNN(f,*args,**kwargs):
    def layer(x, scope='', is_training=True):
        return f(x,*args,**kwargs)
    return layer

# Dropout function. note that is_training is used here
def Dropout(p, name='Dropout'):
    def dropout_layer(x, is_training=True):
        with tf.variable_scope(name,None,[x]):
            # def drop(): return tf.nn.dropout(x,p)
            # def no_drop(): return x
            # return tf.cond(is_training, drop, no_drop)
            if is_training:
                return tf.nn.dropout(x,p)
            else:
                return x
    return dropout_layer

# RELU function.
def ReLU(name='ReLU'):
    def layer(x, is_training=True):
        with tf.variable_scope(name,None,[x]):
            return tf.nn.relu(x)
    return layer

# HardTanh. no need to need to calculate exp(-x), which is costly
def HardTanh(name='HardTanh'):
    def layer(x, is_training=True):
        with tf.variable_scope(name,None,[x]):
            return tf.clip_by_value(x,-1,1)
    return layer

# Reshape the variable
def View(shape, name='View'):
    with tf.variable_scope(name,None,[x], reuse=reuse):
        return wrapNN(tf.reshape,shape=shape)

# MaxPooling function
def SpatialMaxPooling(kW, kH=None, dW=None, dH=None, padding='VALID',
            name='SpatialMaxPooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def max_pool(x,is_training=True):
        with tf.variable_scope(name,None,[x]):
              return tf.nn.max_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return max_pool

# AvgPooling function
def SpatialAveragePooling(kW, kH=None, dW=None, dH=None, padding='VALID',
        name='SpatialAveragePooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def avg_pool(x,is_training=True):
        with tf.variable_scope(name, None, [x]):
              return tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return avg_pool

# Normalization
def BatchNormalization(*kargs, **kwargs):
    return wrapNN(tf.contrib.layers.batch_norm, *kargs, **kwargs)
def InstanceNormalization(*kargs, **kwargs):
    return wrapNN(tf.contrib.layers.instance_norm, *kargs, **kwargs)

# A sequence of layers
def Sequential(moduleList):
    def model(x, is_training=True):
    # Create model
        output = x
        #with tf.variable_scope(name,None,[x]):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
    return model

# concate a sequence of ouputs of layers along axis (=3 as default)
def Concat(moduleList, dim=3):
    def model(x, is_training=True):
    # Create model
        outputs = []
        for i,m in enumerate(moduleList):
            name = 'layer_'+str(i)
            with tf.variable_scope(name, 'Layer', [x], reuse=reuse):
                outputs[i] = m(x, is_training=is_training)
            # 有可能写反了
            # should it be:
            # output = tf.concat(outputs, dim)
            output = tf.concat(dim, outputs)
        return output
    return model


# Layer1 -> [output1]-> Layer2 --[--> output2 + output1] ---> Layer3 
#               |                                   ^
#               V                                   |
#               -------------------------------------
# Layer2 is moduleList
def Residual(moduleList, name='Residual'):
    m = Sequential(moduleList)
    def model(x, is_training=True):
    # Create model
        with tf.variable_scope(name,None,[x]):
            output = tf.add(m(x, is_training=is_training), x)
            return output
    return model

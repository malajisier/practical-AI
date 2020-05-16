import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D, Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from functools import wraps

from utils import compose
from darknet53 import darknet53


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    conv_kwargs = {'kernel_regularizer': l2(5e-4)}  
    conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    conv_kwarg.update(kwargs)
    return Conv2D(*args, **conv_kwargs)

# DBL
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': Fasle}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
    )


def make_last_layers(x, num_filters, out_filters):
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    # 调整通道
    y = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)

    return x, y


# 三个特征层 --> 输出 
def yolo_body(inputs, num_anchors, num_classes):
    feat1, feat2, feat3 = darknet53(inputs)
    darknet = Model(inputs, feat3)

    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))
    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2)(x))
    x = Concatenate()([x, feat2])

    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))
    x = compose(DarknetConv2D(128, (1, 1)), UpSampling2D(2)(x))
    x = compose()([x, feat1])

    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def yolo_head(feats, anchors, num_classes, input_shape, clac_loss = False):
    num_anchors= len(anchors)
    anchors = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 
    # [height, width]
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop = grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop = grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate(grid_x, grid_y)
    grid = K.cast(grid, K.dtype(feats))

    # feats_shape: [batch_size, 13, 13, 3, class+5]
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调整为 真实值
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_hw = K.exp(feats[..., 2:4]) * anchors / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    class_probs = K.sigmoid(feats[..., 5:])

    if clac_loss == True:
        return grid, feats, box_xy, box_hw
    
    return box_xy, box_hw, box_confidence, class_probs


from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D, Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from functools import wraps

from utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    conv_kwargs.update(kwargs)
    return Conv2D(*args, **conv_kwargs)


# 卷积块DBL：conv + bn + leakyRelU
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))


# 类残差块
def resblock(x, num_filters, num_blocks):
    x = ZeroPadding2D((1, 0), (1, 0))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides = (2, 2))(x)

    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(y)
        x = Add()[x, y]

    return x


# feat1: 13x13
# feat2: 26x26 
# feat3: 52x52
def darknet53(x):
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock(x, 64, 1)
    x = resblock(x, 128, 2)
    x = resblock(x, 256, 8)
    feat1 = x
    x = resblock(x, 512, 8)
    feat2 = x
    x = resblock(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3
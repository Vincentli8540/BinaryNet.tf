# TTQ_cifar10.py
from nnUtils import *

model = Sequential([
    SpatialConvolution(128,3,3,1,1, padding='VALID', bias=False,name='conv2d_1'),
    BatchNormalization(),
    HardTanh(),
    TernarizedSpatialConvolution(128,3,3, padding='SAME', bias=False,name='ttq_conv2d_1'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    HardTanh(),
    TernarizedSpatialConvolution(256,3,3, padding='SAME', bias=False,name='ttq_conv2d_2'),
    BatchNormalization(),
    HardTanh(),
    TernarizedSpatialConvolution(256,3,3, padding='SAME', bias=False,name='ttq_conv2d_3'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    HardTanh(),
    TernarizedSpatialConvolution(512,3,3, padding='SAME', bias=False,name='ttq_conv2d_4'),
    BatchNormalization(),
    HardTanh(),
    TernarizedSpatialConvolution(512,3,3, padding='SAME', bias=False,name='ttq_conv2d_5'),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    HardTanh(),
    TernarizedAffine(1024, bias=False),
    BatchNormalization(),
    HardTanh(),
    TernarizedAffine(1024, bias=False),
    BatchNormalization(),
    HardTanh(),
    TernarizedAffine(10),
    BatchNormalization()
])

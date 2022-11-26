from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, LeakyReLU, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from utils.utils import compose

from nets.CSPdarknet53 import darknet_body
from nets.densenet import DenseNet
from nets.ghostnet import Ghostnet
from nets.mobilenet_v3 import MobileNetV3
from nets.efficientnet import EfficientNetB3
from nets.vgg19 import VGG19


#--------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#   测试中发现没有l2正则化效果更好，所以去掉了l2正则化
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   进行五次卷积
#---------------------------------------------------#
def make_five_convs(x, num_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x

#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):    # inputs = Input(shape=(None, None, 3)); num_anchors = 3; num_classes = 20
    BBone = 0

    if BBone == 0:
        # ---------------------------------------------------#
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   feat1 ： 52,52,256   ——> P3 ：52,52,128   ——> P3_output ：52,52,256  ——> P3_output ：52,52,[num_anchors*(num_classes+5)]
        #   feat2 ： 26,26,512   ——> P4 ：26,26,256   ——> P4_output ：26,26,512  ——> P4_output ：26,26,[num_anchors*(num_classes+5)]
        #   feat3 ： 13,13,1024  ——> P5 ：13,13,512   ——> P5_output ：13,13,1024 ——> P5_output ：13,13,[num_anchors*(num_classes+5)]
        # ---------------------------------------------------#
        feat1, feat2, feat3 = darknet_body(inputs)
    elif BBone == 1:
        #---------------------------------------------------#
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        backbone = "densenet121"  # backbone = ["densenet121", "densenet169", "densenet201"]
        feat1, feat2, feat3 = DenseNet(inputs, backbone)
    elif BBone == 2:
        #---------------------------------------------------#
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1, feat2, feat3 = Ghostnet(inputs)
    elif BBone == 3:
        #---------------------------------------------------#
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1, feat2, feat3 = MobileNetV3(inputs, alpha=1.0, kernel=5, se_ratio=0.25)
    elif BBone == 4:
        #---------------------------------------------------#
        #   52,52,48；26,26,136；13,13,384
        #---------------------------------------------------#
        feats, filters_outs = EfficientNetB3(inputs = inputs)
        feat1 = feats[2]
        feat2 = feats[4]
        feat3 = feats[6]
    elif BBone == 5:
        feat1, feat2, feat3 = VGG19(inputs)

    else:
        raise ValueError('Backbone 加载有错误！！！')


    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    # 使用了SPP结构，即不同尺度的最大池化后堆叠。
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])  # -> 13,13,2048  , 4个512堆起来就是2048
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)    # -> 13,13,512

    # 13,13,512 -> 13,13,256 -> 26,26,256
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(P5)
    # 26,26,512 -> 26,26,256
    P4 = DarknetConv2D_BN_Leaky(256, (1,1))(feat2)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P4, P5_upsample])
    
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4,256)   # -> 26,26,256

    # 26,26,256 -> 26,26,128 -> 52,52,128
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(P4)
    # 52,52,256 -> 52,52,128
    P3 = DarknetConv2D_BN_Leaky(128, (1,1))(feat1)
    # 52,52,128 + 52,52,128 -> 52,52,256
    P3 = Concatenate()([P3, P4_upsample])

    # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    P3 = make_five_convs(P3,128)
    
    #---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    #---------------------------------------------------#
    P3_output = DarknetConv2D_BN_Leaky(256, (3,3))(P3)
    P3_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P3_output)

    # 52,52,128 -> 26,26,256
    P3_downsample = ZeroPadding2D(((1,0),(1,0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2))(P3_downsample)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P3_downsample, P4])
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4,256)
    
    #---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    P4_output = DarknetConv2D_BN_Leaky(512, (3,3))(P4)
    P4_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P4_output)
    
    # 26,26,256 -> 13,13,512
    P4_downsample = ZeroPadding2D(((1,0),(1,0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))(P4_downsample)
    # 13,13,512 + 13,13,512 -> 13,13,1024
    P5 = Concatenate()([P4_downsample, P5])
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = make_five_convs(P5,512)
    
    #---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    P5_output = DarknetConv2D_BN_Leaky(1024, (3,3))(P5)
    P5_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(P5_output)

    # inputs = Tensor("input_1:0", shape=(None, None, None, 3), dtype=float32)
    # [P5_output, P4_output, P3_output] = [<tf.Tensor 'conv2d_109/Identity:0' shape=(None, None, None, 75) dtype=float32>, <tf.Tensor 'conv2d_101/Identity:0' shape=(None, None, None, 75) dtype=float32>, <tf.Tensor 'conv2d_93/Identity:0' shape=(None, None, None, 75) dtype=float32>]
    return Model(inputs, [P5_output, P4_output, P3_output])


# ---------------------------------------------------井水不犯河水--------------------------------- 下面的主要用于预测时


#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    #---------------------------------------------------#
    feats = tf.convert_to_tensor(feats)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    #---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    #---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,25)
    #   25可拆分成4 + 1 + 20
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #   返回的 feats.shape=(1, 13, 13, 3, 25)
    #---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    #---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    #---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


#---------------------------------------------------#
#   (针对加了灰条的情况)对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    #-----------------------------------处理图像大小开始-------------------------------
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))  # 返回的 new_shape = tf.Tensor([416. 416.], shape=(2,), dtype=float32)
    # 反正是 image_shape 中较长的一条边先到达 input_shape(416*416), 较短的一条边会小于416
    #-----------------------------------------------------------------#
    #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    #   new_shape指的是宽高缩放情况
    #-----------------------------------------------------------------#
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    #-------------------------------------处理图像大小结束------------------------------
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


#---------------------------------------------------#
#   获取每个box和它的得分
#---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy : -1,13,13,3,2; 
    #   box_wh : -1,13,13,3,2; 
    #   box_confidence : -1,13,13,3,1; 
    #   box_class_probs : -1,13,13,3,80;
    #-----------------------------------------------------------------#
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    #-----------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
    #   因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对齐进行修改，去除灰条的部分。
    #   将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #-----------------------------------------------------------------#
    if letterbox_image:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        boxes =  K.concatenate([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ])
    #-----------------------------------------------------------------#
    #   获得最终得分和框的位置
    #-----------------------------------------------------------------#
    boxes = K.reshape(boxes, [-1, 4])      #  boxes.shape=(507, 4)   507 = 1*13*13*3
    box_scores = box_confidence * box_class_probs      #  box_scores.shape=(1, 13, 13, 3, 20)
    box_scores = K.reshape(box_scores, [-1, num_classes])   #  box_scores.shape=(507, 20)
    return boxes, box_scores

# ---------------------------------------------------#
#   图片预测
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,    # yolo_outputs = [*yolo_model.output, input_image_shape]   ; yolo_model.output即P3P4P5； input_image_shape = [[1330. 1330.]]
              anchors,
              num_classes,
              image_shape,
              max_boxes=1000,
              score_threshold=.6,
              iou_threshold=.5,
              eager = False,
              letterbox_image=True):
    if eager:
        image_shape = K.reshape(yolo_outputs[-1],[-1])    # 输入图片的大小，返回的 image_shape =  tf.Tensor([1330. 1330.], shape=(2,), dtype=float32)
        num_layers = len(yolo_outputs)-1                  # num_layers = 3
    else:
        #---------------------------------------------------#
        #   获得特征层的数量，有效特征层的数量为3
        #---------------------------------------------------#
        num_layers = len(yolo_outputs)

    #-----------------------------------------------------------#
    #   anchors =
    #   [[ 12.  16.]
    #    [ 19.  36.]
    #    [ 40.  28.]
    #    [ 36.  75.]
    #    [ 76.  55.]
    #    [ 72. 146.]
    #    [142. 110.]
    #    [192. 243.]
    #    [459. 401.]]
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    #-----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    #-----------------------------------------------------------#
    #   这里获得的是模型中输入图片的大小，一般是416x416
    #-----------------------------------------------------------#
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32    # input_shape = tf.Tensor([416 416], shape=(2,), dtype=int32)
    boxes = []
    box_scores = []
    #-----------------------------------------------------------#
    #   对每个特征层进行处理
    #-----------------------------------------------------------#
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                    image_shape, letterbox_image)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # l=0时，_boxes.shape=(507, 4);   _box_scores.shape=(507, 20)
    # l=1时，_boxes.shape=(2028, 4);   _box_scores.shape=(2028, 20)
    # l=2时，_boxes.shape=(8112, 4);   _box_scores.shape=(8112, 20)
    # 因此，boxes是最终含有三个数组tensor的列表；box_scores也是最终含有三个数组tensor的列表；
    #-----------------------------------------------------------#
    #   将每个特征层的结果进行堆叠
    #-----------------------------------------------------------#
    boxes = K.concatenate(boxes, axis=0)        # 返回的 boxes.shape = (10647, 4) , 其中 10647 = 507+2028+8112
    box_scores = K.concatenate(box_scores, axis=0)    # 返回的 box_scores.shape = (10647, 20) , 其中 10647 = 507+2028+8112

    mask = box_scores >= score_threshold
    '''
    mask = tf.Tensor(
                    [[False False False ... False False False]
                     [False False False ... False False False]
                     [False False False ... False False False]
                     ...
                     [ True  True  True ...  True  True  True]
                     [ True  True  True ...  True  True  True]
                     [ True  True  True ...  True  True  True]], shape=(10647, 20), dtype=bool)
    
    '''
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')   # max_boxes_tensor = tf.Tensor(20, shape=(), dtype=int32)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        #-----------------------------------------------------------#
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        #-----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是
        #   框的位置，得分与种类
        #-----------------------------------------------------------#
        class_boxes = K.gather(class_boxes, nms_index)             # 返回的 class_boxes = tf.Tensor([[ 711.3347  769.1078 1029.1936 1250.2893]], shape=(1, 4), dtype=float32)
        class_box_scores = K.gather(class_box_scores, nms_index)   # 返回的 class_box_scores = tf.Tensor([0.9986758], shape=(1,), dtype=float32)
        classes = K.ones_like(class_box_scores, 'int32') * c       # 返回的 classes = tf.Tensor([0], shape=(1,), dtype=int32)
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_     # 返回的 boxes_.shape=(n, 4); scores_.shape=(n, 1); classes_.shape=(n, 1)   n即最后要画出的框的个数

# if __name__ == '__main__':
#     # inputs = Input(shape=(None, None, 3))
#     inputs = Input(shape=(416, 416, 3))
#     mod = yolo_body(inputs, num_anchors = 3, num_classes = 20)    # inputs = Input(shape=(None, None, 3)); num_anchors = 3; num_classes = 20
#     mod.summary()
#     print(mod.outputs)





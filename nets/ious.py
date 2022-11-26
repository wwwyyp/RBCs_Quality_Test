import math

import tensorflow as tf
from tensorflow.keras import backend as K


def box_ciou(b1, b2):  # b1 = pred_box, b2 = raw_true_box，  pred_box.shape = (m,13,13,3,4)，   raw_true_box.shape = (m,13,13,3,4)
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    '''
    b1_mins = 
             [[ 6.82181299e-01  8.96811366e-01]
              [-2.31963921e+00  6.76489711e-01]
              [ 5.24982691e-01  6.34652615e-01]]]]], shape=(2, 13, 13, 3, 2), dtype=float32)
    '''
    b1_maxes = b1_xy + b1_wh_half
    '''
    b1_maxes = 
              [[1.1998538  1.0109333 ]
               [1.23035    1.2841029 ]
               [1.4604284  1.4232701 ]]]]], shape=(2, 13, 13, 3, 2), dtype=float32)
    '''
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    '''
    b2_mins = 
             [[0. 0.]
              [0. 0.]
              [0. 0.]]]]], shape=(2, 13, 13, 3, 2), dtype=float32)
    '''
    b2_maxes = b2_xy + b2_wh_half
    '''
    b2_maxes = 
              [[0.       0.      ]
               [0.       0.      ]
               [0.       0.      ]]]]], shape=(2, 13, 13, 3, 2), dtype=float32)
    '''

    # 求真实框和预测框所有的iou
    intersect_mins = K.maximum(b1_mins, b2_mins)      # intersect_mins.shape=(2, 13, 13, 3, 2)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)   # intersect_maxes.shape=(2, 13, 13, 3, 2)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    '''
    intersect_wh = 
                  [[0. 0.]
                   [0. 0.]
                   [0. 0.]]]]], shape=(2, 13, 13, 3, 2), dtype=float32)
    '''
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    '''
    intersect_area = 
                    [[0. 0. 0.]
                     [0. 0. 0.]
                     [0. 0. 0.]
                     ...
                     [0. 0. 0.]
                     [0. 0. 0.]
                     [0. 0. 0.]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    '''
    b1_area = 
             [[0.07505657 0.4479331  0.9935263 ]
              [0.05154469 0.47494316 0.8214685 ]
              [0.05207548 1.2003751  0.6748694 ]
              ...
              [0.04303064 1.1973716  0.92902535]
              [0.05050991 0.4550622  1.0530105 ]
              [0.05874694 0.3670893  1.0167521 ]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    '''
    b2_area = 
             [[0. 0. 0.]
              [0. 0. 0.]
              [0. 0. 0.]
              ...
              [0. 0. 0.]
              [0. 0. 0.]
              [0. 0. 0.]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    union_area = b1_area + b2_area - intersect_area
    '''
    union_area = 
                [[7.84906894e-02 3.41730922e-01 9.32762027e-01]
                 [5.00514396e-02 2.38760769e-01 8.19629431e-01]
                 [5.47877476e-02 2.43431613e-01 7.77557433e-01]
                 ...
                 [4.98475917e-02 2.73439735e-01 8.32990289e-01]
                 [4.66575846e-02 2.48128802e-01 8.45842063e-01]
                 [6.05346411e-02 3.10033619e-01 8.64936531e-01]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    iou = intersect_area / K.maximum(union_area,K.epsilon())    # K.epsilon() 返回1个极小的模糊因子，主要是为了让分母不为0
    '''
    iou = 
         [[0. 0. 0.]
          [0. 0. 0.]
          [0. 0. 0.]
          ...
          [0. 0. 0.]
          [0. 0. 0.]
          [0. 0. 0.]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    # 计算中心的差距
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)   # [K.square(b1_xy - b2_xy)].shape=(2, 13, 13, 3, 2)
    '''
    center_distance = 
                     [[8.94490719e-01 9.33518469e-01 9.62848663e-01]
                      [8.83946896e-01 9.39266086e-01 9.85348165e-01]
                      [8.97538483e-01 9.44065750e-01 1.01639903e+00]
                      ...
                      [1.52660656e+00 1.59603691e+00 1.62595236e+00]
                      [1.63564682e+00 1.72038865e+00 1.78336501e+00]
                      [1.81780720e+00 1.89451420e+00 1.94192505e+00]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # 计算对角线距离
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    '''
    enclose_diagonal = 
                      [[ 1.4110694   1.932698    3.1082604 ]
                       [ 1.3615735   1.9570626   3.05098   ]
                       [ 1.3171706   1.9621919   2.9726727 ]
                       ...
                       [ 2.2784004   2.8763416   3.728198  ]
                       [ 2.2951264   2.9174957   4.170158  ]
                       [ 2.3862333   3.0271993   4.3234615 ]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())
    '''
    ciou = 
          [[-0.6521972  -0.4952612  -0.2837896 ]
           [-0.69091725 -0.5181223  -0.31635657]
           [-0.62804574 -0.5101883  -0.3385298 ]
           ...
           [-0.69802356 -0.41807097 -0.4726645 ]
           [-0.712776   -0.08915448 -0.5005433 ]
           [-0.73960346 -0.09916283 -0.49888736]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    v = 4*K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1],K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v
    '''
    ciou = 
          [[-0.95187527 -0.511854   -0.35088107]
           [-0.95868707 -0.5098968  -0.36491105]
           [-1.0225886  -0.49669996 -0.36369556]
           ...
           [-0.9999013  -0.6355803  -0.49523863]
           [-0.98866606 -0.627297   -0.49541822]
           [-0.9999994  -0.62282497 -0.51232386]]]], shape=(2, 13, 13, 3), dtype=float32)
    '''
    ciou = K.expand_dims(ciou, -1)
    ciou = tf.where(tf.math.is_nan(ciou), tf.zeros_like(ciou), ciou)
    '''
    ciou = 
          [[-1.0141634 ]
           [-0.63411826]
           [-0.5119333 ]]
          [[-1.0292183 ]
           [-0.63642156]
           [-0.525391  ]]]]], shape=(2, 13, 13, 3, 1), dtype=float32)
    '''
    return ciou

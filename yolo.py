import colorsys
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : 'logs/Epoch60-Total_Loss2.7312-Val_Loss11.8636——darknet .h5',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/voc_classes.txt',
        "score"             : 0.10,
        "iou"               : 0.3,
        "max_boxes"         : 1000,
        #-------------------------------#
        #   显存比较小可以使用416x416
        #   显存比较大可以使用608x608
        #-------------------------------#
        "model_image_size"  : (416, 416),
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()    # 返回的是列表
        self.anchors = self._get_anchors()      # 返回的是数组
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #   返回的 class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #   [[ 12.  16.]
    #    [ 19.  36.]
    #    [ 40.  28.]
    #    [ 36.  75.]
    #    [ 76.  55.]
    #    [ 72. 146.]
    #    [142. 110.]
    #    [192. 243.]
    #    [459. 401.]]
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        #---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        #---------------------------------------------------------#
        #   载入模型
        #---------------------------------------------------------#
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        '''
        hsv_tuples = 
                    [(0.0, 1.0, 1.0), (0.05, 1.0, 1.0), (0.1, 1.0, 1.0), (0.15, 1.0, 1.0), (0.2, 1.0, 1.0), 
                     (0.25, 1.0, 1.0), (0.3, 1.0, 1.0), (0.35, 1.0, 1.0), (0.4, 1.0, 1.0), (0.45, 1.0, 1.0), 
                     (0.5, 1.0, 1.0), (0.55, 1.0, 1.0), (0.6, 1.0, 1.0), (0.65, 1.0, 1.0), (0.7, 1.0, 1.0), 
                     (0.75, 1.0, 1.0), (0.8, 1.0, 1.0), (0.85, 1.0, 1.0), (0.9, 1.0, 1.0), (0.95, 1.0, 1.0)]
        '''
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        '''
        self.colors = 
                     [(1.0, 0.0, 0.0), (1.0, 0.30000000000000004, 0.0), (1.0, 0.6000000000000001, 0.0), (1.0, 0.8999999999999999, 0.0), (0.7999999999999998, 1.0, 0.0), 
                      (0.5, 1.0, 0.0), (0.20000000000000018, 1.0, 0.0), (0.0, 1.0, 0.09999999999999964), (0.0, 1.0, 0.40000000000000036), (0.0, 1.0, 0.7000000000000002), 
                      (0.0, 1.0, 1.0), (0.0, 0.6999999999999997, 1.0), (0.0, 0.40000000000000036, 1.0), (0.0, 0.09999999999999964, 1.0), (0.1999999999999993, 0.0, 1.0), 
                      (0.5, 0.0, 1.0), (0.8000000000000007, 0.0, 1.0), (1.0, 0.0, 0.9000000000000004), (1.0, 0.0, 0.5999999999999996), (1.0, 0.0, 0.3000000000000007)]
        '''
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        '''
        self.colors = 
                     [(255, 0, 0), (255, 76, 0), (255, 153, 0), (255, 229, 0), (203, 255, 0), 
                      (127, 255, 0), (51, 255, 0), (0, 255, 25), (0, 255, 102), (0, 255, 178), 
                      (0, 255, 255), (0, 178, 255), (0, 102, 255), (0, 25, 255), (50, 0, 255), 
                      (127, 0, 255), (204, 0, 255), (255, 0, 229), (255, 0, 152), (255, 0, 76)]
        '''
        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        '''
        self.colors = 
                     [(0, 178, 255), (255, 153, 0), (51, 255, 0), (0, 102, 255), (255, 76, 0), 
                      (204, 0, 255), (255, 0, 76), (203, 255, 0), (127, 0, 255), (0, 255, 178), 
                      (255, 229, 0), (0, 255, 255), (0, 25, 255), (0, 255, 102), (255, 0, 229), 
                      (0, 255, 25), (127, 255, 0), (50, 0, 255), (255, 0, 152), (255, 0, 0)]
        '''
        np.random.seed(None)
        #---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        #---------------------------------------------------------#
        self.input_image_shape = Input([2,],batch_size=1)    # self.input_image_shape = Tensor("input_1:0", shape=(1, 2), dtype=float32)
        inputs = [*self.yolo_model.output, self.input_image_shape]   #  self.yolo_model.output[0].shape=(None, None, None, 75)
        outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
            arguments={'anchors': self.anchors, 'num_classes': len(self.class_names), 'image_shape': self.model_image_size,
            'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes, 'letterbox_image': self.letterbox_image})(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        # 返回的outputs包含这三个部分： boxes_.shape=(n, 4); scores_.shape=(n, 1); classes_.shape=(n, 1)
 
    @tf.function
    def get_pred(self, image_data, input_image_shape):  # image_data.shape = (1, 416, 416, 3)    #  input_image_shape = [[315. 332.]]  input_image_shape.shape = (1, 2)
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes   # 返回的 out_boxes.shape=boxes_.shape=(n, 4); out_scores.shape=scores_.shape=(n, 1); out_classes.shape=classes_.shape=(n, 1)

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #---------------------------------------------------------#
        image = image.convert('RGB')
        
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:  # 加灰条，不扭曲
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:   # 不加灰条，极可能扭曲
            boxed_image = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.   image_data.shape = (1, 416, 416, 3)
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0) #  input_image_shape = [[315. 332.]]  input_image_shape.shape = (1, 2)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)
        # 返回的 out_boxes.shape=boxes_.shape=(n, 4); out_scores.shape=scores_.shape=(n, 1); out_classes.shape=classes_.shape=(n, 1)   ， n即最后要画出的框的个数
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #---------------------------------------------------------#
        #   设置字体
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)     # image 为实际待识别的图片大小。
        
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        # image.save("./红细胞质量检测结果forpaper/每个图片检测结果效果图/21-result_0.60.png")
        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

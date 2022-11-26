from tensorflow.keras import backend, layers
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, LeakyReLU, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D, Input)




def VGG19(inputs):
    # 第一层 (None, 416, 416, 64)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)

    # 第二层(None, 208, 208, 64)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 第三层(None, 208, 208, 128)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # 第四层(None, 104, 104, 128)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # 第五层(None, 104, 104, 256)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # 第六层(None, 104, 104, 256)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # 第七层(None, 104, 104, 256)
    x = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    # 第八层（新增卷积层3*3*256） (None, 52, 52, 256)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    feat1 = x

    # 第九层 (None, 52, 52, 512)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # 第十层 (None, 52, 52, 512)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # 第十一层 (None, 52, 52, 512)
    x = layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    # 第十二层（新增卷积层3*3*512）  (None, 26, 26, 512)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    feat2 = x

    # 第十三层 (None, 26, 26, 512)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # 第十四层 (None, 26, 26, 512)
    x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # 第十五层 (None, 26, 26, 512)
    x = layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    # 第十六层（新增卷积层3*3*512） (None, 13, 13, 1024)
    x = layers.Conv2D(1024, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    feat3 = x

    return feat1, feat2, feat3


# inputs = Input(shape=(416, 416, 3))
# feat1, feat2, feat3 = VGG19(inputs)
# print(feat1)
# print(feat2)
# print(feat3)






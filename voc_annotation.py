#---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
train_sets=[('2007', 'train')]
val_sets=[('2007', 'val')]

#-----------------------------------------------------#
#   这里设定的classes顺序要和model_data里的txt一样
#-----------------------------------------------------#
classes = ["SDC", "CDC", "CDD", "CSD", "CSE", "SSE", "SVSE"]
train_ji_shu = [0, 0, 0, 0, 0, 0, 0, 0]
val_ji_shu = [0, 0, 0, 0, 0, 0, 0, 0]
# print("初始值  第一类有：{}个；  第二类有：{}个；  第三类有：{}个；  第四类有：{}个；  第五类有：{}个；  第六类有：{}个；  第七类有：{}个；  错误个数有：{}个。".format(ji_shu[0], ji_shu[1], ji_shu[2], ji_shu[3], ji_shu[4],ji_shu[5],ji_shu[6], ji_shu[7]))


def convert_annotation(year, image_id, list_file, ji_shu):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)

        if cls_id==0:
            ji_shu[0]=ji_shu[0]+1
        elif cls_id==1:
            ji_shu[1]=ji_shu[1]+1
        elif cls_id==2:
            ji_shu[2]=ji_shu[2]+1
        elif cls_id==3:
            ji_shu[3]=ji_shu[3]+1
        elif cls_id==4:
            ji_shu[4]=ji_shu[4]+1
        elif cls_id==5:
            ji_shu[5]=ji_shu[5]+1
        elif cls_id==6:
            ji_shu[6]=ji_shu[6]+1
        else:
            ji_shu[7]=ji_shu[7]+1

        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in train_sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set), encoding='utf-8').read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.png'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file, train_ji_shu)
        list_file.write('\n')
    list_file.close()

print("训练集中 第一类有：{}个；  第二类有：{}个；  第三类有：{}个；  第四类有：{}个；  第五类有：{}个；  第六类有：{}个；  第七类有：{}个；  错误个数有：{}个。".format(train_ji_shu[0], train_ji_shu[1], train_ji_shu[2], train_ji_shu[3], train_ji_shu[4],train_ji_shu[5],train_ji_shu[6], train_ji_shu[7]))



for year, image_set in val_sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set), encoding='utf-8').read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.png'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file, val_ji_shu)
        list_file.write('\n')
    list_file.close()

print("验证集中 第一类有：{}个；  第二类有：{}个；  第三类有：{}个；  第四类有：{}个；  第五类有：{}个；  第六类有：{}个；  第七类有：{}个；  错误个数有：{}个。".format(val_ji_shu[0], val_ji_shu[1], val_ji_shu[2], val_ji_shu[3], val_ji_shu[4], val_ji_shu[5], val_ji_shu[6], val_ji_shu[7]))


























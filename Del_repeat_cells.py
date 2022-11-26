import tensorflow as tf
import numpy as np
import sys
# sys.setrecursionlimit(3000)

def bubble_sort(out_classes0, out_scores0, out_boxes0):
    for i in range(len(out_scores0) - 1):  # 0-18
        for j in range(len(out_scores0) - 1):  # 0-18
            if (out_scores0[j] < out_scores0[j + 1]):
                tmps = out_scores0[j].copy();
                out_scores0[j] = out_scores0[j + 1].copy();
                out_scores0[j + 1] = tmps
                tmpc = out_classes0[j].copy();
                out_classes0[j] = out_classes0[j + 1].copy();
                out_classes0[j + 1] = tmpc
                tmpb = out_boxes0[j].copy();
                out_boxes0[j] = out_boxes0[j + 1].copy();
                out_boxes0[j + 1] = tmpb
    return out_classes0, out_scores0, out_boxes0


def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou



def panduan(list1, list2):
    top1, left1, bottom1, right1 = list1
    top2, left2, bottom2, right2 = list2
    box_1 = [left1, top1, right1, bottom1]
    box_2 = [left2, top2, right2, bottom2]

    list1_xm = (left1 + right1) / 2
    list1_ym = (top1 + bottom1) / 2
    list2_xm = (left2 + right2) / 2
    list2_ym = (top2 + bottom2) / 2

    xm_d = list1_xm-list2_xm
    ym_d = list1_ym-list2_ym
    # d0 = list1[0]-list2[0]
    # d1 = list1[1]-list2[1]
    # d2 = list1[2]-list2[2]
    # d3 = list1[3]-list2[3]
    # # if ((abs(d0)<=5) or (abs(d1)<=5) or (abs(d2)<=5) or (abs(d3)<=5)):
    if ((abs(xm_d) <= 10) and (abs(ym_d) <= 10)) or cal_iou(box_1, box_2)>0.3:
    # if ((abs(d0) <= 5) or (abs(d1) <= 5) or (abs(d2) <= 5) or (abs(d3) <= 5)) and (abs(xm_d) <= 5) and (abs(ym_d) <= 5):
        return True
    else:
        return False



def delete_repeat_cells(out_boxes_in, out_scores_in, out_classes_in):
    box_array = out_boxes_in.numpy()
    score_array = out_scores_in.numpy()
    name_array = out_classes_in.numpy()

    boxes_list = []
    scores_list = []
    classes_list = []

    def putin(out_boxesw, out_scoresw, out_classesw):
        out_boxes = out_boxesw; out_scores = out_scoresw; out_classes = out_classesw
        lenth = len(out_boxes)
        if len(out_boxes) == 0:
            print("已成空列表！！！")
        else:
            boxes_list.append(out_boxes[0])
            scores_list.append(out_scores[0])
            classes_list.append(out_classes[0])
            box = out_boxes[0].copy()
            for i in range(lenth):
                tf = panduan(box, out_boxes[i])
                if tf:
                    out_boxes[i] = ''; out_scores[i] = ''; out_classes[i] = ''
                else:
                    continue
            while '' in out_boxes:
                 out_boxes.remove('')
                 out_scores.remove('')
                 out_classes.remove('')
            putin(out_boxes, out_scores, out_classes)


    out_classes1, out_scores1, out_boxes1 = bubble_sort(name_array, score_array, box_array)
    out_boxes2 = out_boxes1.tolist()
    out_classes2 = out_classes1.tolist()
    out_scores2 = out_scores1.tolist()

    putin(out_boxes2, out_scores2, out_classes2)

    box_array_back = np.array(boxes_list)
    score_array_back = np.array(scores_list)
    name_array_back = np.array(classes_list)

    box_tensor_back = tf.convert_to_tensor(box_array_back)
    score_tensor_back = tf.convert_to_tensor(score_array_back)
    name_tensor_back = tf.convert_to_tensor(name_array_back)

    return box_tensor_back, score_tensor_back, name_tensor_back


# box = [[88, 309, 144, 368],
#        [92, 307, 141, 365]]
# score = [0.2101, 0.3104]
# name = [1, 0]
#
# box_array = np.array(box)
# score_array = np.array(score)
# name_array = np.array(name)
#
# box_tensor = tf.convert_to_tensor(box_array)
# score_tensor = tf.convert_to_tensor(score_array)
# name_tensor = tf.convert_to_tensor(name_array)
#
# a, b, c = delete_repeat_cells(box_tensor, score_tensor, name_tensor)
#
# print(a)
# print(b)
# print(c)


import numpy as np
import random
import cv2


def nms(predicts_dict, threshold):
    # 对每一个类别分别进行NMS；一次读取一对键值（即某个类别的所有框）
    for object_name, bbox in predicts_dict.items():
        bbox_array = np.array(bbox, dtype=np.float)
        x1 = bbox_array[:, 0]
        y1 = bbox_array[:, 1]
        x2 = bbox_array[:, 2]
        y2 = bbox_array[:, 3]
        scores = bbox_array[:, 4]

        # 按置信度大小降序排序，计算每个检测框的面积
        order = scores.argsort()[::-1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        # 按confidence从高到低遍历bbx，移除所有与该矩形框的IoU值大于threshold的矩形框
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算最大置信度检测框与剩余检测框的相交区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 若不相交，宽高为0
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # 注意这里都是采用广播机制
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= threshold)[0]
            # 将order中的第inds+1处的值重新赋值给order；即更新保留下来的索引，加1是因为因为没有计算与自身的IOU
            order = order[inds + 1]

        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()
    return predicts_dict


# 下面在一张全黑图片上测试非极大值抑制的效果
img = np.zeros((600, 600), np.uint8)
predicts_dict = {'black1': [[83, 54, 165, 163, 0.8], [67, 48, 118, 132, 0.5], [91, 38, 192, 171, 0.6]],
                 'black2': [[59, 120, 137, 368, 0.12], [54, 154, 148, 382, 0.13]]}

# 在全黑的图像上画出设定的几个框
for object_name, bbox in predicts_dict.items():
    for box in bbox:
        x1, y1, x2, y2, score = box[0], box[1], box[2], box[3], box[-1]
        # uniform()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。
        # uniform() 方法将随机生成下一个实数，它在 [x, y) 范围内
        y_text = int(random.uniform(y1, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))

    # 创建一个显示图像的窗口
    cv2.namedWindow("black1_roi")
    # 在窗口中显示图像;注意这里的窗口名字如果不是刚创建的窗口的名字，则会自动创建一个新的窗口并将图像显示在这个窗口
    cv2.imshow("black1_roi", img)
    cv2.waitKey(0)  # 如果不添这一句，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过。
    cv2.destroyAllWindows()

# 在全黑图片上画出经过非极大值抑制后的框
img_cp = np.zeros((600, 600), np.uint8)
predicts_dict_nms = nms(predicts_dict, 0.1)
for object_name, bbox in predicts_dict_nms.items():
    for box in bbox:
        x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[-1]
        y_text = int(random.uniform(y1, y2))
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img_cp, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))

    cv2.namedWindow("black1_nms")
    cv2.imshow("black1_nms", img_cp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
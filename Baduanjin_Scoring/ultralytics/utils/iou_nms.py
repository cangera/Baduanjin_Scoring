import numpy as np
import torch


def py_nms(boxes, threshold):
    """Pure Python NMS baseline."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # 计算每一个anchor的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按照从小到大排序后返回下标，然后顺序取反，即从大到小对应的下标
    # order = scores.argsort()[::-1]
    order = torch.argsort(scores, descending=True)  # bounding box 的置信度排序
    keep = []
    while order.size(0) > 0:
        i = order[0]  # 置信度最高的 bounding box 的索引
        keep.append(int(i))  # 添加本次置信度最高的 bounding box 的索引

        # 当前 bbox 和剩下的 bbox 之间的交叉区域
        # 选择大于 x1, y1 和小于 x2, y2 的区域
        xx1 = torch.max(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
        yy1 = torch.max(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
        xx2 = torch.min(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
        yy2 = torch.min(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

        # 当前 bbox 和其他剩下的 bbox 之间交叉区域的面积
        w = torch.clamp(xx2 - xx1 + 1, min=0.0)
        h = torch.clamp(yy2 - yy1 + 1, min=0.0)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留交集小于一定阈值的 bounding box
        inds = torch.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return boxes[keep]

def py_nms_val(allboxes, threshold):
    """Pure Python NMS baseline."""
    l = len(allboxes)
    for j in range(l):
        boxes = allboxes[j]
        if len(boxes) == 0:
            continue
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        # 计算每一个anchor的面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按照从小到大排序后返回下标，然后顺序取反，即从大到小对应的下标
        # order = scores.argsort()[::-1]
        order = torch.argsort(scores, descending=True)  # bounding box 的置信度排序
        keep = []
        while order.size(0) > 0:
            i = order[0]  # 置信度最高的 bounding box 的索引
            keep.append(int(i))  # 添加本次置信度最高的 bounding box 的索引

            # 当前 bbox 和剩下的 bbox 之间的交叉区域
            # 选择大于 x1, y1 和小于 x2, y2 的区域
            xx1 = torch.max(x1[i], x1[order[1:]])  # 交叉区域的左上角的横坐标
            yy1 = torch.max(y1[i], y1[order[1:]])  # 交叉区域的左上角的纵坐标
            xx2 = torch.min(x2[i], x2[order[1:]])  # 交叉区域右下角的横坐标
            yy2 = torch.min(y2[i], y2[order[1:]])  # 交叉区域右下角的纵坐标

            # 当前 bbox 和其他剩下的 bbox 之间交叉区域的面积
            w = torch.clamp(xx2 - xx1 + 1, min=0.0)
            h = torch.clamp(yy2 - yy1 + 1, min=0.0)
            inter = w * h

            # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留交集小于一定阈值的 bounding box
            inds = torch.where(ovr <= threshold)[0]
            order = order[inds + 1]
            allboxes[j] = boxes[keep]
    return allboxes
# if __name__ == "__main__":
#     a = np.array([[191, 89, 413, 420, 0.80],      # 0
#                   [281, 152, 573, 510, 0.99],     # 1
#                   [446, 294, 614, 471, 0.65],     # 2
#                   [50, 453, 183, 621, 0.98],      # 3
#                   [109, 474, 209, 635, 0.78]])    # 4
#     nms_result = py_nms(a, 0.2)
#     print(nms_result)


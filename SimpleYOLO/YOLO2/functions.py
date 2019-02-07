import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon import HybridBlock

from SimpleYOLO.YOLO2.utils import transform_center, transform_size


def yolo2_decoder(x, num_class, anchor_scales):
    """
    yolo2_decoder 会把卷积的通道分开，转换，最后转成我们需要的检测框
    out: (index,score,xmin,ymin,xmax,ymax)
    """
    stride = num_class + 5
    x = x.transpose((0, 2, 3, 1))  # (Batch,H,W,Stride*Anchor)
    x = x.reshape((0, 0, 0, -1, stride))  # (Batch,H,W,Anchor,Stride)

    xy_pred = x.slice_axis(begin=0, end=2, axis=-1)
    wh = x.slice_axis(begin=2, end=4, axis=-1)
    score_pred = x.slice_axis(begin=4, end=5, axis=-1)
    cls_pred = x.slice_axis(begin=5, end=stride, axis=-1)

    xy = nd.sigmoid(xy_pred)
    x, y = transform_center(xy)
    w, h = transform_size(wh, anchor_scales)
    score = nd.sigmoid(score_pred)
    cid = nd.argmax(cls_pred, axis=-1, keepdims=True)

    left = nd.clip(x - w / 2, 0, 1)
    top = nd.clip(y - h / 2, 0, 1)
    right = nd.clip(x + w / 2, 0, 1)
    bottom = nd.clip(y + h / 2, 0, 1)
    output = nd.concat(*[cid, score, left, top, right, bottom], dim=4)
    return output, cls_pred, score, nd.concat(*[xy, wh], dim=4)


def yolo2_target(scores, output, labels, anchors, ignore_label=-1, thresh=0.5):
    """
    定义一个函数来生成yolo2训练目标
    YOLO2寻找真实目标的方法比较特殊，是在每个格点内各自比较，而不是使用全局的预设。
    这里我们使用了一个技巧：sample_weight（个体权重）矩阵， 用于损失函数内部权重的调整，
    我们也可以通过权重矩阵来控制哪些个体需要被屏蔽，这一点在目标检测中尤其重要，因为往往大多数的背景区域不需要预测检测框。

    网络预测的输出为 (32,16,16,2,5)
    而label的形式为：labels 即 ground truth(32,1,5)，其中 5 包括一个class label:0，以及左上、右下两个corner相对于整张图的坐标
    模型回归的目标形式：
    """
    b, h, w, n, _ = scores.shape
    anchors = np.reshape(np.array(anchors), (-1, 2))
    """ 这里传入scores只是为了用其shape和context
    scores = nd.slice_axis(outputs, begin=1, end=2, axis=-1)
    boxes = nd.slice_axis(outputs, begin=2, end=6, axis=-1)
    gt_boxes = nd.slice_axis(labels, begin=1, end=5, axis=-1)
    """
    target_score = nd.zeros((b, h, w, n, 1), ctx=scores.context)
    target_id = nd.ones_like(target_score, ctx=scores.context) * ignore_label
    target_box = nd.zeros((b, h, w, n, 4), ctx=scores.context)
    sample_weight = nd.zeros((b, h, w, n, 1), ctx=scores.context)
    for b in range(output.shape[0]):
        # find the best match for each ground-truth
        label = labels[b].asnumpy()
        valid_label = label[np.where(label[:, 0] > -0.5)[0], :]
        # shuffle because multi gt could possibly match to one anchor, we keep the last match randomly
        np.random.shuffle(valid_label)
        for l in valid_label:
            gx, gy, gw, gh = (l[1] + l[3]) / 2, (l[2] + l[4]) / 2, l[3] - l[1], l[4] - l[2]
            ind_x = int(gx * w)
            ind_y = int(gy * h)
            tx = gx * w - ind_x
            ty = gy * h - ind_y
            gw = gw * w
            gh = gh * h
            # find the best match using width and height only, assuming centers are identical
            intersect = np.minimum(anchors[:, 0], gw) * np.minimum(anchors[:, 1], gh)
            ovps = intersect / (gw * gh + anchors[:, 0] * anchors[:, 1] - intersect)
            best_match = int(np.argmax(ovps))
            target_id[b, ind_y, ind_x, best_match, :] = l[0]
            target_score[b, ind_y, ind_x, best_match, :] = 1.0
            tw = np.log(gw / anchors[best_match, 0])
            th = np.log(gh / anchors[best_match, 1])
            target_box[b, ind_y, ind_x, best_match, :] = mx.nd.array([tx, ty, tw, th])
            sample_weight[b, ind_y, ind_x, best_match, :] = 1.0
            # print('ind_y', ind_y, 'ind_x', ind_x, 'best_match', best_match, 't', tx, ty, tw, th, 'ovp', ovps[best_match], 'gt', gx, gy, gw/w, gh/h, 'anchor', anchors[best_match, 0], anchors[best_match, 1])
    return target_id, target_score, target_box, sample_weight


class YOLO2Output(HybridBlock):
    """
    YOLO2Output 包括一个卷积映射
    不包含解码器
    """

    def __init__(self, num_class, anchor_scales, **kwargs):
        super(YOLO2Output, self).__init__(**kwargs)
        assert num_class > 0, "number of classes should > 0, given {}".format(num_class)
        self._num_class = num_class
        assert isinstance(anchor_scales, (list, tuple)), "list or tuple of anchor scales required"
        assert len(anchor_scales) > 0, "at least one anchor scale required"
        for anchor in anchor_scales:
            assert len(anchor) == 2, "expected each anchor scale to be (width, height), provided {}".format(anchor)
        self._anchor_scales = anchor_scales
        out_channels = len(anchor_scales) * (num_class + 1 + 4)
        with self.name_scope():
            self.output = nn.Conv2D(out_channels, 1, 1)

    def hybrid_forward(self, F, x, *args):
        return self.output(x)

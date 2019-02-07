import time

import mxnet as mx
import mxnet.gluon.nn as nn
import numpy as np
from mxnet import autograd, gluon, image, nd

from SimpleYOLO.YOLO2.metric import LossRecorder
from SimpleYOLO.YOLO2.functions import YOLO2Output, yolo2_decoder, yolo2_target
from projects.pikachu.config import Config

batch_size = 32
ctx = mx.gpu(0)

data_h, data_w = Config.size
class_names = Config.classes
num_class = len(Config.classes)
anchors = Config.anchors
train_data = image.ImageDetIter(
    data_shape=(3, data_h, data_w),
    std=np.array([255, 255, 255]),
    path_imgrec=Config.train_rec,
    path_imgidx=Config.train_idx,
    batch_size=batch_size,
    shuffle=True,
    rand_mirror=True,
    rand_crop=1,
    min_object_covered=0.95,
    max_attempts=100,
    num_thread=4
)
test_data = image.ImageDetIter(
    data_shape=(3, data_h, data_w),
    std=np.array([255, 255, 255]),
    path_imgrec=Config.val_rec,
    batch_size=batch_size,
    shuffle=False
)

sce_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
l1_loss = gluon.loss.L1Loss()
obj_loss = LossRecorder('objectness_loss')
cls_loss = LossRecorder('classification_loss')
box_loss = LossRecorder('box_refine_loss')
positive_weight = 5.0
negative_weight = 0.1
class_weight = 1.0
box_weight = 5.0

print('ready to train')

net = nn.HybridSequential()
sym = gluon.nn.SymbolBlock.imports(
    Config.backbone_symbol,
    [Config.backbone_input],
    param_file=Config.backbone_params,
    ctx=ctx
)
net.add(sym)

# use 2 classes, 1 as dummy class, otherwise softmax won't work
predictor = YOLO2Output(2, anchors)
predictor.initialize()
net.add(predictor)

print('initialized')
net.collect_params().reset_ctx(ctx)
# 'sgd', {'learning_rate': 0.01, 'wd': 5e-4}
trainer = gluon.Trainer(net.collect_params(), 'adam')
net.hybridize()

round = 50
now = time.strftime("%Y%m%d%H%M%S", time.localtime())
log_file = open("logs/%s.log" % now, 'w+')
for epoch in range(round):
    # reset data iterators and metrics
    train_data.reset()
    cls_loss.reset()
    obj_loss.reset()
    box_loss.reset()
    tic = time.time()
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            x = net(x)
            output, cls_pred, score, xywh = yolo2_decoder(x, 2, anchors)
            with autograd.pause():
                tid, tscore, tbox, sample_weight = yolo2_target(score, xywh, y, anchors, thresh=0.5)
            # losses
            loss1 = sce_loss(cls_pred, tid, sample_weight * class_weight)
            score_weight = nd.where(
                sample_weight > 0,
                nd.ones_like(sample_weight) * positive_weight,
                nd.ones_like(sample_weight) * negative_weight
            )
            loss2 = l1_loss(score, tscore, score_weight)
            loss3 = l1_loss(xywh, tbox, sample_weight * box_weight)
            loss = loss1 + loss2 + loss3
        loss.backward()
        trainer.step(batch_size)
        # update metrics
        cls_loss.update(loss1)
        obj_loss.update(loss2)
        box_loss.update(loss3)

    log = (
        epoch + 1,
        cls_loss.get()[0],
        cls_loss.get()[1],
        obj_loss.get()[0],
        obj_loss.get()[1],
        box_loss.get()[0],
        box_loss.get()[1],
        time.time() - tic
    )
    string = 'Epoch %2d, train %s %f, %s %.5f, %s %.5f time %.1f sec' % log
    print(string)
    print(string, file=log_file)

print('finished')

# net.save_parameters('models/yolo_pikachu-0050.params')
net.export('models/yolo_pikachu', round)

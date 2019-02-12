import time

import mxnet as mx
import mxnet.gluon.nn as nn
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.data import DataLoader
from mxnet.gluon.model_zoo import vision

from SimpleYOLO.YOLO2.functions import YOLO2Output, yolo2_decoder, yolo2_target
from SimpleYOLO.YOLO2.metric import LossRecorder
from .data.config import Config
from .data.dataset import VOCDataset

batchsize = 4
ctx = mx.gpu(0)

train_dataset = VOCDataset(
    annotation_dir=Config.annotation_dir,
    img_dir=Config.img_dir,
    dataset_index=Config.dataset_index,
)

train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

sce_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
l1_loss = gluon.loss.L1Loss()

obj_loss = LossRecorder('objectness_loss')
cls_loss = LossRecorder('classification_loss')
box_loss = LossRecorder('box_refine_loss')

positive_weight = 5.0
negative_weight = 0.1
class_weight = 1.0
box_weight = 5.0

pretrained = vision.get_model('resnet34_v2', pretrained=True).features
net = nn.HybridSequential()

for i in range(len(pretrained) - 2):
    net.add(pretrained[i])

# anchor scales, try adjust it yourself
scales = [
    [3.3004, 3.59034],
    [9.84923, 8.23783]
]

predictor = YOLO2Output(21, Config.classes)
predictor.initialize()
net.add(predictor)
net.collect_params().reset_ctx(ctx)

trainer = gluon.Trainer(
    net.collect_params(),
    'sgd',
    {'learning_rate': 0.1, 'wd': 5e-4}
)

for epoch in range(100):
    # reset data iterators and metrics
    # train_data.reset()
    cls_loss.reset()
    obj_loss.reset()
    box_loss.reset()
    tic = time.time()
    for iteration, (data, label) in enumerate(train_dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        batch_size = data.shape[0]
        with autograd.record():
            x = net(data)
            output, cls_pred, score, xywh = yolo2_decoder(x, 21, scales)
            with autograd.pause():
                tid, tscore, tbox, sample_weight = yolo2_target(score, xywh, label, scales, thresh=0.5)
            # losses
            loss1 = sce_loss(cls_pred, tid, sample_weight * class_weight)
            score_weight = nd.where(sample_weight > 0,
                                    nd.ones_like(sample_weight) * positive_weight,
                                    nd.ones_like(sample_weight) * negative_weight)
            loss2 = l1_loss(score, tscore, score_weight)
            loss3 = l1_loss(xywh, tbox, sample_weight * box_weight)
            loss = loss1 + loss2 + loss3
        loss.backward()
        # print(loss)
        trainer.step(batch_size)
        # update metrics
        cls_loss.update(loss1)
        obj_loss.update(loss2)
        box_loss.update(loss3)
    logs = (
        epoch,
        cls_loss.get()[0],
        cls_loss.get()[1],
        obj_loss.get()[0],
        obj_loss.get()[1],
        box_loss.get()[0],
        box_loss.get()[1],
        time.time() - tic
    )
    print('Epoch %2d, train %s %f, %s %.5f, %s %.5f time %.1f sec' % logs)

net.collect_params().save('gluon.params')

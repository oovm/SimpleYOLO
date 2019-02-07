import os

import mxnet as mx
import mxnet.gluon.nn as nn
import numpy as np
from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet.gluon.model_zoo import vision

from SimpleYOLO.YOLO2.metric import LossRecorder
from SimpleYOLO.YOLO2.functions import YOLO2Output, yolo2_decoder, yolo2_target

root_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/'
data_dir = 'data/pikachu/'
dataset = {
    'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
    'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
    'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'
}
for k, v in dataset.items():
    print(data_dir + k)
    if not os.path.exists(data_dir + k):
        gluon.utils.download(root_url + k, data_dir + k, sha1_hash=v)

data_shape = 256
batch_size = 32
rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])


def get_iterators(data_shape, batch_size):
    class_names = ['pikachu', 'dummy']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir + 'train.rec',
        path_imgidx=data_dir + 'train.idx',
        shuffle=True,
        mean=True,
        std=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200
    )
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir + 'val.rec',
        shuffle=False,
        mean=True,
        std=True
    )
    return train_iter, val_iter, class_names, num_class


train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)

batch = train_data.next()
print(batch)

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=linewidth)


'''
_, figs = plt.subplots(3, 3, figsize=(6,6))
for i in range(3):
    for j in range(3):
        img, labels = batch.data[0][3*i+j], batch.label[0][3*i+j]
        img = img.transpose((1, 2, 0)) * rgb_std + rgb_mean
        img = img.clip(0,255).asnumpy()/255
        fig = figs[i][j]
        fig.imshow(img)
        for label in labels:
            rect = box_to_rect(label[1:5]*data_shape,'red',2)
            fig.add_patch(rect)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
plt.show()
'''

sce_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
l1_loss = gluon.loss.L1Loss()

obj_loss = LossRecorder('objectness_loss')
cls_loss = LossRecorder('classification_loss')
box_loss = LossRecorder('box_refine_loss')

positive_weight = 5.0
negative_weight = 0.1
class_weight = 1.0
box_weight = 5.0

pretrained = vision.get_model('resnet18_v1', pretrained=True).features
net = nn.HybridSequential()

for i in range(len(pretrained) - 2):
    net.add(pretrained[i])

# anchor scales, try adjust it yourself
scales = [
    [3.3004, 3.59034],
    [9.84923, 8.23783]
]

# use 2 classes, 1 as dummy class, otherwise softmax won't work
predictor = YOLO2Output(2, scales)
predictor.initialize()
net.add(predictor)

ctx = mx.gpu(0)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(
    net.collect_params(),
    'sgd', {'learning_rate': 0.01, 'wd': 5e-4}
)

import time
from mxnet import autograd

for epoch in range(20):
    # reset data iterators and metrics
    train_data.reset()
    cls_loss.reset()
    obj_loss.reset()
    box_loss.reset()
    tic = time.time()
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        print(y)
        with autograd.record():
            x = net(x)
            output, cls_pred, score, xywh = yolo2_decoder(x, 2, scales)
            with autograd.pause():
                tid, tscore, tbox, sample_weight = yolo2_target(score, xywh, y, scales, thresh=0.5)
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

    print('Epoch %2d, train %s , %s , %s  time %.1f sec' % (
        epoch, cls_loss.get()[0], obj_loss.get()[0], box_loss.get()[0], time.time() - tic))

net.collect_params().save('gluon.params')


def process_image(fname):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())
    # resize to data_shape
    data = image.imresize(im, data_shape, data_shape)
    # minus rgb mean, divide std
    data = (data.astype('float32') - rgb_mean) / rgb_std
    # convert to batch x channel x height xwidth
    return data.transpose((2, 0, 1)).expand_dims(axis=0), im


def predict(x):
    x = net(x)
    output, cls_prob, score, xywh = yolo2_decoder(x, 2, scales)
    return nd.contrib.box_nms(output.reshape((0, -1, 6)))


x, im = process_image('pika.jpg')
out = predict(x.as_in_context(ctx))

mpl.rcParams['figure.figsize'] = (6, 6)

colors = ['blue', 'green', 'red', 'black', 'magenta']


def display(im, out, threshold=0.5):
    plt.imshow(im.asnumpy())
    for row in out:
        row = row.asnumpy()
        class_id, score = int(row[0]), row[1]
        if class_id < 0 or score < threshold:
            continue
        color = colors[class_id % len(colors)]
        box = row[2:6] * np.array([im.shape[0], im.shape[1]] * 2)
        rect = box_to_rect(nd.array(box), color, 2)
        plt.gca().add_patch(rect)
        text = class_names[class_id]
        plt.gca().text(box[0], box[1],
                       '{:s} {:.2f}'.format(text, score),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=10, color='white')
    plt.show()


display(im, out[0], threshold=0.5)

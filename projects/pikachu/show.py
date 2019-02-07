import matplotlib as mpl
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from mxnet import gluon, image, nd

from SimpleYOLO.YOLO2.functions import yolo2_decoder
from projects.pikachu.config import Config

ctx = mx.gpu()
data_h, data_w = 256, 256
anchors = Config.anchors
class_names = Config.classes

mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['figure.figsize'] = (6, 6)

net = gluon.nn.SymbolBlock.imports(
    'models/yolo_pikachu-symbol.json',
    ['data'],
    param_file='models/yolo_pikachu-0050.params',
    ctx=ctx
)

print('ready to show')


def process_image(file_name):
    with open(file_name, 'rb') as f:
        im = image.imdecode(f.read())
    # resize to data_shape
    data = image.imresize(im, data_h, data_w)
    # minus rgb mean, divide std
    data = data.astype('float32') / 255
    # convert to batch x channel x height xwidth
    return data.transpose((2, 0, 1)).expand_dims(axis=0), im


def predict(x):
    x = net(x)
    output, cls_prob, score, xywh = yolo2_decoder(x, 2, anchors)
    return nd.contrib.box_nms(output.reshape((0, -1, 6)))


x, im = process_image('data/pika.jpg')
out = predict(x.as_in_context(ctx))

colors = ['blue', 'green', 'red', 'black', 'magenta']


def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False,
        edgecolor=color,
        linewidth=linewidth
    )


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
        plt.gca().text(
            box[0], box[1],
            '{:s} {:.2f}'.format(text, score),
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=10, color='white'
        )
    plt.show()


display(im, out[0], threshold=0.8)

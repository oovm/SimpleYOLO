import os

from mxnet import gluon

root_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/'
dataset = {
    'pikachu-train.rec': root_url + 'train.rec',
    'pikachu-train.idx': root_url + 'train.idx',
    'pikachu-val.rec': root_url + 'val.rec'
}
for k, v in dataset.items():
    print(k)
    if not os.path.exists(k):
        gluon.utils.download(v, k)

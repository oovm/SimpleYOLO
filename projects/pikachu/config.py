class Config:
    version = "yolo2"  # Model name
    backbone_input = 'Input'  # Input Port Name
    backbone_symbol = 'external/resnet18_V1-symbol.json'  # Symbol Path
    backbone_params = 'external/Resnet18_V1-0000.params'  # Array Path

    train_idx = 'data/pikachu-train.idx'
    train_rec = 'data/pikachu-train.rec'
    val_rec = 'data/pikachu-val.rec'

    classes = ['pikachu', 'dummy']

    size = [256, 256]
    rgb_mean = [0.482353, 0.458824, 0.407843]
    rgb_std = [0.229, 0.224, 0.225]
    anchors = [
        [3.3004, 3.59034],
        [9.84923, 8.23783]
    ]

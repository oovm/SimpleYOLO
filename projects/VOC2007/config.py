class Config:
    version = "yolo2"  # Model name
    input = "data"  # Input Port Name
    backbone_symbol = ""  # Symbol Path
    backbone_params = ""  # Array Path

    train_idx = "./train.idx"
    train_rec = "./train.rec"
    val_rec = "./val.rec"

    classes = [
        "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle",
        "Bus", "Car", "Cat", "Chair", "Cow",
        "Dining Table", "Dog", "Horse", "Motorbike", "Person",
        "Potted Plant", "Sheep", "Sofa", "Train", "Tv/Monitor"
    ]

    size = [320, 240]
    rgb_mean = [0.482353, 0.458824, 0.407843]
    rgb_std = [0.229, 0.224, 0.225]
    anchors = [
        [1.08, 1.19],
        [3.42, 4.41],
        [6.63, 11.38],
        [9.42, 5.11],
        [16.62, 10.52]
    ]

from mxnet import nd


def transform_center(xy):
    """
    Given x, y prediction after sigmoid(), convert to relative coordinates (0, 1) on image.
    xy: (Batch,H,W,Anchor,2<HalfCoordinates>)
    """
    b, h, w, n, s = xy.shape
    x_array = nd.arange(0, w, repeat=(n * 1), ctx=xy.context).reshape((1, 1, w, n, 1))
    y_array = nd.arange(0, h, repeat=(w * n * 1), ctx=xy.context).reshape((1, h, w, n, 1))

    offset_x = nd.tile(x_array, (b, h, 1, 1, 1))
    # print(offset_x[0].asnumpy()[:, :, 0, 0])
    offset_y = nd.tile(y_array, (b, 1, 1, 1, 1))
    # print(offset_y[0].asnumpy()[:, :, 0, 0])

    x, y = xy.split(num_outputs=2, axis=-1)
    x = (x + offset_x) / w
    y = (y + offset_y) / h
    return x, y


def transform_size(wh, anchors):
    """
    Given w, h prediction after exp() and anchor sizes, convert to relative width/height (0, 1) on image
    wh: (Batch,H,W,Anchor,2(Coordinates))
    anchors: (Anchor,2)
    """
    b, h, w, n, s = wh.shape
    anchor_array = nd.array(anchors, ctx=wh.context).reshape((1, 1, 1, -1, 2))
    aw, ah = nd.tile(anchor_array, (b, h, w, 1, 1)).split(num_outputs=2, axis=-1)
    w_pred, h_pred = nd.exp(wh).split(num_outputs=2, axis=-1)
    w_out = w_pred * aw / w
    h_out = h_pred * ah / h
    return w_out, h_out


def corner2center(boxes, concat=True):
    """
    Convert left/top/right/bottom style boxes into x/y/w/h format
    """
    left, top, right, bottom = boxes.split(axis=-1, num_outputs=4)
    x = (left + right) / 2
    y = (top + bottom) / 2
    width = right - left
    height = bottom - top
    if concat:
        last_dim = len(x.shape) - 1
        return nd.concat(*[x, y, width, height], dim=last_dim)
    return x, y, width, height


def center2corner(boxes, concat=True):
    """
    Convert x/y/w/h style boxes into left/top/right/bottom format
    """
    x, y, w, h = boxes.split(axis=-1, num_outputs=4)
    w2 = w / 2
    h2 = h / 2
    left = x - w2
    top = y - h2
    right = x + w2
    bottom = y + h2
    if concat:
        last_dim = len(left.shape) - 1
        return nd.concat(*[left, top, right, bottom], dim=last_dim)
    return left, top, right, bottom

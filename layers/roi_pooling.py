import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable as tVariable

has_backward = True

def roi_pooling_pytorch(input, rois, size=(7, 7), spatial_scale=1.0):  # pytorch version use for loop !!!
    assert rois.dim() == 2
    assert rois.size(1) == 5
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        #im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        im = input.narrow(0, im_idx,1)
        im = im[...,roi[2]:(roi[4]+1),roi[1]:(roi[3]+1)]
        output.append(F.adaptive_max_pool2d(im, size))

    output = torch.cat(output, 0)
    if has_backward:
        #        output.backward(output.data.clone())
        output.sum().backward()
    return output

if __name__ == "__main__":
    # batch_size, img_size, num_rois
    config = [[1, 50, 300], [8, 8, 100],
              [64, 64, 100], [64, 64, 1000],
              [256, 256, 100], [256, 256, 1000]]
    T = 5
    cuda = True
    has_backward = True

    print('use_cuda: {}, has_backward: {}'.format(cuda, has_backward))
    for i in range(len(config)):
        x = torch.rand((config[i][0], 512, config[i][1], config[i][1]))
        rois = torch.rand((config[i][2], 5))
        rois[:, 0] = rois[:, 0] * config[i][0]
        rois[:, 1:] = rois[:, 1:] * config[i][1]
        for j in range(config[i][2]):
            max_, min_ = max(rois[j, 1], rois[j, 3]), min(rois[j, 1], rois[j, 3])
            rois[j, 1], rois[j, 3] = min_, max_
            max_, min_ = max(rois[j, 2], rois[j, 4]), min(rois[j, 2], rois[j, 4])
            rois[j, 2], rois[j, 4] = min_, max_
        rois = torch.floor(rois)
        x = tVariable(x, requires_grad=True)
        rois = tVariable(rois, requires_grad=False)

        if cuda:
            x = x.cuda()
            rois = rois.cuda()

        start = time.time()
        for t in range(T):
            output = roi_pooling_pytorch(x, rois)
        print('method{}: {}, batch_size: {}, size: {}, num_rois: {}'.format(0, (time.time() - start) / T, config[i][0],config[i][1],config[i][2]))
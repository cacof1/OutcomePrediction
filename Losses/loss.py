##Losses                                                                                                                                                                                                                          
import torch
import numpy as np


def CrossEntropy(output, target):
    log_prob = F.log_softmax(output, dim=1)
    loss = F.nll_loss(log_prob, torch.argmax(target, dim=1), reduction='none')
    return torch.mean(loss)


def SoftDiceLoss(output, target):
    """
   Reference: Milletari, F., Navab, N., & Ahmadi, S. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric                                                                                                            
   Medical Image Segmentation. In International Conference on 3D Vision (3DV).                                                                                                                                                    
   """
    output = F.logsigmoid(output).exp()
    axes = list(range(2, len(output.shape)))
    eps = 1e-10
    intersection = torch.sum(output * target + eps, axes)
    output_sum_square = torch.sum(output * output + eps, axes)
    target_sum_square = torch.sum(target * target + eps, axes)
    sum_squares = output_sum_square + target_sum_square
    return 1.0 - 2.0 * torch.mean(intersection / sum_squares)


def WeightedMSE(prediction, labels, weights=None, label_range=None):
    loss = 0
    for i, label in enumerate(labels):
        idx = (label_range == int(label.cpu().numpy())).nonzero()
        if (idx is not None) and (idx[0][0] < 60):
            loss = loss + (prediction[i] - label) ** 2 * weights[idx[0][0]]
        else:
            loss = loss + (prediction[i] - label) ** 2 * weights[-1]
    loss = loss / (i + 1)
    return loss

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = (outputs.squeeze(1) > 0.5).int()
    labels = (labels > 0.5).int()
    intersection = torch.sum((outputs & labels).float())
    union = torch.sum((outputs | labels).float())

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

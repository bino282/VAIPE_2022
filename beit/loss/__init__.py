import logging
from re import L
from loss.focal_loss import FocalLoss
from loss.lovasz_loss import LovaszSoftmax
from loss.ohem_loss import OhemCrossEntropy2d
from loss.softiou_loss import SoftIoULoss
from loss.recall_loss import RecallCrossEntropy
from loss.balanced_loss import BalancedCrossEntropy
from loss.dice_loss import DiceLoss
import torch
from loss.asl import *
from loss.cce import CCE

def get_loss_function(loss_type,n_classes,**kwargs):
    if 1:
        if loss_type == 'CrossEntropy' or loss_type == 'BalancedCE' or loss_type == 'WeightedCE' :
            criterion = torch.nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            criterion = FocalLoss(**kwargs)
        elif loss_type == 'Lovasz':
            criterion = LovaszSoftmax(**kwargs)
        elif loss_type == 'OhemCrossEntropy':
            criterion = OhemCrossEntropy2d(**kwargs)
        elif loss_type == 'SoftIOU':
            criterion = SoftIoULoss(n_classes,**kwargs)
        elif loss_type == 'RecallCE':
            criterion = RecallCrossEntropy(n_classes,**kwargs)
        elif loss_type == 'Diceloss':
            criterion = DiceLoss()
        elif loss_type == 'ASL':
            criterion = ASLSingleLabel()
        elif loss_type == 'CCE':
            criterion = CCE(**kwargs)
        else:
            raise NotImplementedError
        print("Using {} ".format(loss_type))
        return criterion

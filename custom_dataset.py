import os
import sys
import numpy as np

from PIL import Image,ImageOps
import torch

from torchvision import transforms
# from _temp.getsize import getsize


# Create custom train params  pool
def create_train_params_pool(args):
    raise NotImplementedError

def get_custom_dataset(name, data_dir):
    '''
    input:
        data_dir:

    output:
        X_tr: 
            torch tensor containing all labeled frames
            shape:(num_LB_samples, height, width)
            dtype:torch.uint8
        Y_tr:
            torch tensor containing all labels of the training split
            shape:(num_LB_samples)
            dtype:torch.int64
        X_te:
            torch tensor containing all unlabeled frames
            shape:(num_UL_samples, height, width)
            dtype:torch.uint8
        Y_te:
            torch tensor containing all labels from the unlabeled pool
            shape:(num_UL_samples)
            dtype:torch.int64
    '''
    raise NotImplementedError
    # X_tr, Y_tr, X_te, Y_te=None,None,None,None
    # return X_tr, Y_tr, X_te, Y_te


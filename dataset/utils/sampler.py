'''sampler for pytorch dataloader, to deal with data imbalance'''

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def test_data_sampler(dataloader, print_batch=False):
    '''
    test function for testing train_loader and see if we are sampling evenly
    NOTE: this is only used for two classes
    '''
    num_class_zero = 0
    num_class_one = 1
    for i, (data, target) in enumerate(dataloader):
        class_zero = len(np.where(target.numpy() == 0)[0])
        class_one = len(np.where(target.numpy() == 1)[0])
        if print_batch:
            print("batch index {}, 0/1: {}/{}".format(i, class_zero, class_one))
        num_class_zero += class_zero
        num_class_one += class_one
    print(
        f'There are {num_class_zero} from class 0 and {num_class_one} from class 1')


def get_sampler(target: list, idx_train=None):
    '''
    ref: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/5

    we also support using the sampler with train test split. this is bc for WeightedRandomSampler it needs to know 

    @params:
      target: list of [0,0,0,0,1,1,1,1,1] 
        where at index 0 has n_classes[0] number of sample in class 0 and so on
    '''
    if len(target) == 0:
        return None
    n_classes = np.unique(target, return_counts=True)[1]
    n = np.sum(n_classes)
    weight = 1. / n_classes
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()

    # WeightedRandomSampler needs to know for each elem of the array, the weight for that elem
    # and for each elem in we need to assign the weight. the weight depends the class we use 1/count for that
    # class and it just works
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler

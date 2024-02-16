from collections import defaultdict
import glob
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch
from dataset.dataset.dataset_ssl_base import Dataset_ssl_base
from typing import Any
import sys
sys.path.append("/../../Self-Supervised-Learning/")


class EyeTracking_Dataset(Dataset_ssl_base):
    def __init__(self,
                 sequence,
                 labels,
                 labels_expert,
                 labels_feedback,
                 transform=None,
                 transform_ssl=None,
                 mode='ssl',
                 triplet_mode='online',
                 labels_postive=None,
                 allCurrFixName=None
                 ) -> None:
        '''
        @params:
            triplet_mode: we can choose what pairs to use during online (within epcoh) or offline. online requres 
            labels: glaucoma label
            labels_postive: used when triplet_mode == 'online'; a list of class index, labels_positive[i] == labels_positive[j] means they are from the same class
                            For example, 'EEM16' is now 15 for example
        '''
        super().__init__(transform, transform_ssl, mode)
        self.seq = sequence
        self.labels = labels
        self.triplet_mode = triplet_mode
        self.labels_postive = labels_postive
        self.labels_expert = labels_expert
        self.labels_feedback = labels_feedback
        self.idx_path = defaultdict(int)

        self.idx_path = {i: allCurrFixName[i] for i in range(
            len(allCurrFixName))}  # defaultdict(str)

        assert len(labels_feedback) == len(labels_expert)

    def __len__(self):
        return len(self.seq)

    def get_positive_label(self, idx):
        '''return label needed depending on the mode'''
        if self.triplet_mode == 'online':
            return self.labels_postive[idx]
        elif self.triplet_mode == 'offline':
            return

    def __getitem__(self, idx: Any) -> Any:
        if self.mode == 'test':
            self.test_idx.append(idx)
        if self.mode == 'ssl':
            # return ancher sequence (h,); (num positive for current anchor,); (1,)
            return (idx, -1), (self.seq[idx]), (int(self.labels[idx]), int(self.labels_expert[idx]), int(self.labels_feedback[idx])), (self.get_positive_label(idx))

        return (idx, -1), (self.seq[idx]), (int(self.labels[idx]), int(self.labels_expert[idx]), int(self.labels_feedback[idx])), (self.get_positive_label(idx))

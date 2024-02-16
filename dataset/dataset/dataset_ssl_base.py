from torch.utils.data import Dataset
import warnings
from typing import Any
import torch


'''
We want to have a standardized way of handling our data, load each img as they are being used to train
this is to prevent caching all imgs in RAM and running out of space

For each dataset object we need to have
    - training/val/test file names
    - label dict[file name] = 'label'
    - path dict[file name] = 'path'
'''


class Dataset_ssl_base(Dataset):
    def __init__(self,
                 transform=None,
                 transform_ssl=None,
                 mode='ssl',
                 ) -> None:
        '''
        @params
            - reports: [] of report used for training/val/testing
            - label: dict label[report_name] = int, class label
            - path: dict report_path[report_name] = path
            - transform: transform for the images to right sizes/preprocessing
            - transform_ssl: transform for ssl to get different views of same image
            - test_mode: True if we are using this dataset to test, store order of img seen
        '''
        super().__init__()
        self.transform, self.transform_ssl = transform, transform_ssl
        self.mode = mode

        if self.mode == 'test':
            self.test_idx = []

        # printing statistics
        # self.dataset_stats(self.labels)

    def dataset_stats(self, labels):
        # this is to print how num sample in each class for ex
        warnings.warn('dataset_stats not implemented!')

    def configure_mode(self, mode):
        if mode == 'test':
            self.test_idx = []
        self.mode = mode

    def __len__(self):
        warnings.warn('Implement this!')

    def __getitem__(self, idx: Any) -> Any:
        warnings.warn('Implement this!')

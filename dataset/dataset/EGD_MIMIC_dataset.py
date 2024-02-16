from dataset.utils.util_read import read_image
import torch
import pandas as pd
import numpy as np
from dataset.dataset.dataset_ssl_base import Dataset_ssl_base
from typing import Any
import sys
sys.path.append("/../../Self-Supervised-Learning/")


EGD_MIMIC_CLASS_MAPPING = {
    'Normal': 0,
    'CHF': 1,
    'pneumonia': 1
}

SPECIAL_TOKENS = {
    'pad_index': 0,
    'cls_index': 1,
    'eos_index': 2
}


def pad_inputs(data, max_seq_length):
    # Concatenate the tensors
    result_tensor = torch.cat(
        [
            torch.tensor([SPECIAL_TOKENS['cls_index']]),
            # shift gaze word and clip them, -2 to account for cls and eos token
            data[:max_seq_length-2] + len(SPECIAL_TOKENS),
            torch.tensor([SPECIAL_TOKENS['eos_index']]),
            torch.zeros(max_seq_length - \
                        len(data[:max_seq_length-2]), dtype=torch.int)
        ]
    )
    # trim incase it is longer than max seq length
    return result_tensor[:max_seq_length]


class EGD_MIMIC_Dataset(Dataset_ssl_base):
    def __init__(self,
                 imgID=[],
                 df_master=None,
                 data_type='imgNgaze',
                 max_seq_length=512,
                 posPairs_criteria='online',
                 transform=None,
                 transform_ssl=None,
                 mode='ssl'
                 ) -> None:
        '''
        get gaze and/or image, not grabbing the one that is empty 

        @params
            - transform_ssl: transform for ssl to get different views of same image
            - mode: 'ssl', 'sl' or 'test'; if test then store order of img seen
            - data_type: 'imgNgaze', 'img', 'gaze'
        '''
        super().__init__()
        self.imgID = imgID
        self.df_master = df_master
        self.max_seq_length = max_seq_length
        self.posPairs_criteria = posPairs_criteria
        self.data_type = data_type

        # actual index to path
        self.idx_path = {
            idx: self.df_master.loc[imgid, 'path'] for idx, imgid in enumerate(self.imgID)}
        self.labels = {
            idx: EGD_MIMIC_CLASS_MAPPING[self.df_master.loc[imgid, 'class']] for idx, imgid in enumerate(self.imgID)}
        # image id to label
        self.imgid2labels = {
            imgid: EGD_MIMIC_CLASS_MAPPING[self.df_master.loc[imgid, 'class']] for imgid in self.imgID}

        self.transform, self.transform_ssl = transform, transform_ssl
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []

    def configure_mode(self, mode):
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []

    def __len__(self):
        return len(self.imgID)

    def read_seq(self, p):
        df = pd.read_csv(p)
        # pad and replace the column with padded values
        return pad_inputs(torch.tensor(df['word_idx'].values), self.max_seq_length)

    def __getitem__(self, idx: Any) -> Any:
        '''
        1. get the image
        2. preprocess report with self.transform
        3. sample two images with self.transform_ssl
        '''
        if self.mode == 'test':
            self.test_idx.append(idx)
        img_path = self.df_master.loc[self.imgID[idx], 'path']
        gaze_path = self.df_master.loc[self.imgID[idx], 'gaze_path']

        x_img = -1 if not (self.data_type == 'imgNgaze' or self.data_type ==
                           'img') else read_image(img_path)
        x_gaze = -1 if not (self.data_type == 'imgNgaze' or self.data_type ==
                            'gaze') else self.read_seq(gaze_path)

        y_label = self.imgid2labels[self.imgID[idx]]

        meta = (idx, gaze_path)
        data = (x_gaze)
        label = (y_label, -1, -1)
        pos_label = (y_label)

        if self.data_type == 'gaze':
            return meta, data, label, pos_label

        meta = (idx, img_path, gaze_path)

        if self.mode == 'sl' or self.mode == 'test':
            if self.transform:
                x_img = self.transform(x_img)
            return meta, (x_img/255.0, x_gaze), (y_label)
        elif self.mode == 'ssl':
            x_img_i = self.transform_ssl(x_img)/255.0
            x_img_j = self.transform_ssl(x_img)/255.0
            return meta, (x_img_i, x_img_j, x_gaze), (y_label)

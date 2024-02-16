from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict
import glob
from typing import Any
import numpy as np
import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from dataset.utils.util_gaze import helper_draw_heatmap
from torchvision.io import read_image
import cv2
from torchvision import transforms


# last update 7.5.2023
expertise = {
    'EEM1': 'removed for anonymity',
    'EEM3': 'removed for anonymity',
    'EEM4': 'removed for anonymity',
    'EEM5': 'removed for anonymity',
    'EEM6': 'removed for anonymity',
    'EEM7': 'removed for anonymity',
    'EEM8': 'removed for anonymity',
    'EEM9': 'removed for anonymity',
    'EEM10': 'removed for anonymity',
    'EEM11': 'removed for anonymity',
    'EEM12': 'removed for anonymity',
    'EEM13': 'removed for anonymity',
    'EEM14': 'removed for anonymity',
    'EEM15': 'removed for anonymity',
    'EEM17': 'removed for anonymity'
}


region_box = {  # works well for this: RLS_038_OD_TC, each one (x, y)
    'circum_rnfl': [(0.058, 0.198), (0.537, 0.422)],
    'en_face': [(0.02, 0.6988), (0.2244, 0.24)],
    'rnfl_thickness': [(0.244, 0.7), (0.222, 0.238)],
    'rnfl_p': [(0.647, 0.22), (0.281, 0.4)],
    'gcl_thickness': [(0.6, 0.685), (0.16, 0.234)],
    'gcl_p': [(0.795, 0.688), (0.16, 0.234)]
}


def build_combined_dataset(save_path_base,
                           report_pavloviaFixation,
                           report_tobiiFixation,
                           pavlovia_path,
                           tobii_path,
                           report_path,
                           data_type='report_eyetrack_combined'
                           ):
    '''
    builds the dataset and save to save_path_base
    ie '/../../data/pavlovia_report_fixation_combined/'

    NOTE: current plt method also downsamples image by 1/2, can change dpi in plt.plot to create dataset with 
    original resolution

    @param:
        data_type: 'report_eyetrack_combined' or 'report_eyetrack'; overlay or save fixation in a separate folder
        report_path: dict[rpt] = path to img
    '''

    combined_paths = glob.glob(f"{save_path_base}*")
    for rpt_n in tqdm(report_pavloviaFixation):
        for fix_n in report_pavloviaFixation[rpt_n]:

            # check if already created skip if combined
            savefilename = f'{save_path_base}{fix_n}'
            if savefilename in combined_paths:
                continue

            fix_n_split = fix_n.split('_')
            split_idx = 0
            for i, w in enumerate(fix_n_split):
                if w == 'fixations':
                    split_idx = i

            folder = '_'.join(fix_n_split[0:split_idx])
            fn = '_'.join(fix_n_split[split_idx:])

            fixation_path = f'{pavlovia_path}{folder}/{fn}.csv'
            try:
                df_fixation = pd.read_csv(fixation_path)
            except:
                print('Something is Wrong :(')
                print('fix_n_split:', fix_n_split)
                print('folder:', folder)
                print('fn:', fn)
                print('fix_n:', fix_n)
                print('\n'*3)
            # print(fixation_path)
            rpt_p = report_path[rpt_n]
            rpt_img = plt.imread(report_path[rpt_n][0])
            height, width, channels = rpt_img.shape

            if data_type == 'report_eyetrack_combined':
                fig = helper_draw_heatmap(
                    df_fixation,
                    (int(width), int(height), int(channels)),
                    imagefile=report_path[rpt_n][0],
                    alpha=0.5,
                    cmap='jet',
                    savefilename=savefilename)
            elif data_type == 'report_eyetrack':
                fig = helper_draw_heatmap(
                    df_fixation,
                    (int(width), int(height), int(channels)),
                    # imagefile= report_path[rpt_n][0] ,
                    alpha=0.5,
                    cmap='jet',
                    savefilename=savefilename)

                return

    for rpt_n in tqdm(report_tobiiFixation):
        for fix_n in report_tobiiFixation[rpt_n]:

            # check if already created skip if combined
            savefilename = f'{save_path_base}{fix_n}'
            if savefilename in combined_paths:
                continue
            folder = fix_n.split('_')[0]
            fn = '_'.join(fix_n.split('_')[1:])

            


class Report_EyeTracking_Dataset(Dataset):
    def __init__(self,
                 reports=None,
                 pavlovia_combined_path=None,
                 pavlovia_data_path=None,
                 report_pavloviaFixation=None,
                 report_label=None,
                 report_path=None,
                 data_type='report_eyetrack_combined',
                 transform=None,
                 transform_ssl=None,
                 mode='ssl',
                 ) -> None:
        '''
        1. assume we have to use fixations with >5 in train for now 


        Example:
            - report_name
            - fixation file name

        @params
            - reports: [] of report used for training
            - pavlovia_combined_path: base path to combined dataset folder
            - pavlovia_data_path: actual path to the data
            - report_pavloviaFixation: dict[rpt_name] = [fixation names]
            - report_label: dict report_label[report_name] = 'G' or 'S'
            - report_path: dict report_path[report_name] = path
                        NOTE: if data_type combined, then its path to combined, if to separate, then path to normal report
            - transform: transform for the images to right sizes/preprocessing
            - transform_ssl: transform for ssl to get different views of same image
            - data_type
        '''
        super().__init__()
        self.reports = reports
        self.pavlovia_combined_path = pavlovia_combined_path
        self.pavlovia_data_path = pavlovia_data_path
        self.report_label = report_label
        self.report_path = report_path
        self.report_pavloviaFixation = report_pavloviaFixation
        self.data_type = data_type

        self.transform = transform
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []
        self.transform_ssl = transform_ssl
        self.idx_path = {
            i: self.report_path[self.reports[i]][0] for i in range(len(self.reports))}

    def configure_mode(self, mode):
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []

    def __len__(self):
        return len(self.reports)

    def read_image(self, fix):

        p = f'{self.pavlovia_combined_path}{fix}.png'
        image = read_image(p)
        # if 4 channels need to convert colour space
        if image.shape[0] != 3:
            image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            image = torch.tensor(image).permute((2, 0, 1))
        return image

    def __getitem__(self, idx: Any) -> Any:
        '''
        1. get the image
        2. preprocess report with self.transform
        3. sample two images with self.transform_ssl
        '''
        rpts, fix = self.reports[idx]
        if self.mode == 'test':
            self.test_idx.append(idx)

        if self.data_type == 'report_eyetrack_combined':
            x = self.read_image(fix)
            if self.transform:
                x = self.transform(x)
            if self.mode == 'sl' or self.mode == 'test':
                return x, torch.tensor(1.0 if self.report_label[rpts] == 'G' else 0.0).long()

            if self.transform_ssl:
                x_i = self.transform_ssl(x)
                x_j = self.transform_ssl(x)
                return x_i, x_j

        elif self.data_type == 'report_eyetrack':
            x = self.read_image(rpts)
            # eye tracking map
            e = torch.load(f'{self.pavlovia_data_path}/{fix}')

            if self.transform:
                x = self.transform(x)
            if self.mode == 'sl' or self.mode == 'test':
                return x, torch.tensor(1.0 if self.report_label[rpts] == 'G' else 0.0).long(), e

            if self.transform_ssl:
                x_i = self.transform_ssl(x)
                x_j = self.transform_ssl(x)
                return x_i, x_j


class Report_Dataset(Dataset):
    def __init__(self,
                 reports=None,
                 report_label=None,
                 ret_rpt_name=False,
                 report_path=None,
                 transform=None,
                 transform_ssl=None,
                 mode='ssl'
                 ) -> None:
        '''
        1. assume we have to use fixations with >5 in train for now 
        2. can be used for both report or report image with eye fixation overlayed
            2.1 for report only: 'pavlovia_combined_path' and 'report_pavloviaFixation' have to be None

        Example:
            - report_name
            - fixation file name

        @params
            - ret_rpt_name: mask used for supcon, true -> then we return rpt name for supcon
            - reports: [] of report used for training
            - report_label: dict report_label[report_name] = 'G' or 'S'
            - report_path: dict report_path[report_name] = path
            - transform: transform for the images to right sizes/preprocessing
            - transform_ssl: transform for ssl to get different views of same image
            - mode: 'ssl', 'sl' or 'test'; if test then store order of img seen
        '''
        super().__init__()

        self.reports = reports

        self.report_label = report_label
        self.report_path = report_path
        self.ret_rpt_name = ret_rpt_name

        self.transform, self.transform_ssl = transform, transform_ssl
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []

        # printing statistics
        self.labels = np.array(
            [1.0 if self.report_label[r] == 'G' else 0 for r in self.reports])
        print(
            f'There are {np.sum(self.labels)} G and {len(self.labels) - np.sum(self.labels) } S ')

        # for use in metrics.py, helper_df
        self.idx_path = {
            i: self.report_path[self.reports[i]][0] for i in range(len(self.reports))}

    def configure_mode(self, mode):
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []


    def __len__(self):
        return len(self.reports)

    def read_image(self, idx):

        p = self.report_path[self.reports[idx]][0]
        image = read_image(p)
        # if 4 channels need to convert colour space
        if image.shape[0] != 3:
            image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            image = torch.tensor(image).permute((2, 0, 1))
        return image

    def __getitem__(self, idx: Any) -> Any:
        '''
        1. get the image
        2. preprocess report with self.transform
        3. sample two images with self.transform_ssl
        '''
        if self.mode == 'test':
            self.test_idx.append(idx)
        rpts = self.reports[idx]
        x = self.read_image(idx)
        y_label = torch.tensor(
            1.0 if self.report_label[rpts] == 'G' else 0.0).long()

        if self.mode == 'sl' or self.mode == 'test':
            if self.transform:
                x = self.transform(x)
            return (idx, '', ''), (x/255.0, 0), (y_label)
        if self.transform_ssl:
            x_i = self.transform_ssl(x)
            x_j = self.transform_ssl(x)
            if self.ret_rpt_name:
                return (idx, rpts, 0), (x_i/255.0, x_j/255.0, 0), (y_label)
            return (idx, 0, 0), (x_i/255.0, x_j/255.0, 0), (y_label)


class Report_Simple_Dataset(Dataset):
    def __init__(self,
                 imgPaths=None,
                 labels=None,
                 transform=None,
                 transform_ssl=None,
                 ret_rpt_name=True,
                 mode='ssl'
                 ) -> None:
        '''

        @params
            - transform_ssl: transform for ssl to get different views of same image
            - mode: 'ssl', 'sl' or 'test'; if test then store order of img seen
        '''
        super().__init__()
        self.imgPaths = imgPaths
        self.labels = labels
        self.ret_rpt_name = ret_rpt_name

        # for use in metrics.py, helper_df
        self.idx_path = {i: self.imgPaths[i]
                         for i in range(len(self.imgPaths))}

        self.transform, self.transform_ssl = transform, transform_ssl
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []

        print(
            f'There are {np.sum(self.labels)} Positive and {len(self.labels) - np.sum(self.labels) } Negative ')

    def configure_mode(self, mode):
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []

    def __len__(self):
        return len(self.imgPaths)

    def read_image(self, p):
        image = read_image(p)
        # if 4 channels need to convert colour space
        if image.shape[0] != 3:
            image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            image = torch.tensor(image).permute((2, 0, 1))
        return image

    def __getitem__(self, idx: Any) -> Any:
        '''
        1. get the image
        2. preprocess report with self.transform
        3. sample two images with self.transform_ssl
        '''
        if self.mode == 'test':
            self.test_idx.append(idx)

        x = self.read_image(self.imgPaths[idx])
        # pytorch loss_fn needs type int
        y_label = self.labels[idx].astype(int)

        if self.mode == 'sl' or self.mode == 'test':
            if self.transform:
                x = self.transform(x)
            # return x/255.0, y_label
            return (idx, '', ''), (x/255.0, 0), (y_label)
        elif self.mode == 'ssl':
            x_i = self.transform_ssl(x)
            x_j = self.transform_ssl(x)
            # return x_i/255.0, x_j/255.0, y_label
            rpts = self.reports[idx]
            if self.ret_rpt_name:
                return (idx, rpts, 0), (x_i/255.0, x_j/255.0, 0), (y_label)
            return (idx, 0, 0), (x_i/255.0, x_j/255.0, 0), (y_label)


class Report_MultiModal_Dataset(Dataset):
    def __init__(self,
                 imgPaths=None,
                 labels=None,
                 transform=None,
                 transform_ssl=None,
                 mode='ssl'
                 ) -> None:
        '''

        @params
            - transform_ssl: transform for ssl to get different views of same image
            - mode: 'ssl', 'sl' or 'test'; if test then store order of img seen
        '''
        super().__init__()
        self.imgPaths = imgPaths
        self.labels = labels
        # for use in metrics.py, helper_df
        self.idx_path = {i: self.imgPaths[i]
                         for i in range(len(self.imgPaths))}

        self.y_size, self.x_size = None, None

        self.transform, self.transform_ssl = transform, transform_ssl
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []
        print(
            f'There are {np.sum(self.labels)} Positive and {len(self.labels) - np.sum(self.labels) } Negative ')

    def configure_mode(self, mode):
        self.mode = mode
        if self.mode == 'test':
            self.test_idx = []

    def __len__(self):
        return len(self.imgPaths)

    def read_image(self, p):
        image = read_image(p)
        # if 4 channels need to convert colour space
        if image.shape[0] != 3:
            image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            image = torch.tensor(image).permute((2, 0, 1))
        return image

    def split_image(self, x, scan_type):
        upper_left, size_box = region_box[scan_type]
        return transforms.functional.crop(x, int(self.y_size*upper_left[1]), int(self.x_size*upper_left[0]), int(self.y_size*size_box[1]),  int(self.x_size*size_box[0]))

    def __getitem__(self, idx: Any) -> Any:
        '''
        1. get the image
        2. preprocess report with self.transform
        3. sample two images with self.transform_ssl
        '''
        if self.mode == 'test':
            self.test_idx.append(idx)

        x = self.read_image(self.imgPaths[idx])/255.0

        if not self.y_size and not self.x_size:
            _, self.y_size, self.x_size = x.shape
        # pytorch loss_fn needs type int
        y_label = self.labels[idx].astype(int)

        x_circum_rnfl = self.split_image(x, 'circum_rnfl')
        x_rnfl_p = self.split_image(x, 'rnfl_p')
        x_en_face = self.split_image(x, 'en_face')
        x_rnfl_thickness = self.split_image(x, 'rnfl_thickness')
        x_gcl_thickness = self.split_image(x, 'gcl_thickness')
        x_gcl_p = self.split_image(x, 'gcl_p')

        if self.mode == 'sl' or self.mode == 'test':
            x = [
                self.transform(x_circum_rnfl),
                self.transform(x_rnfl_p),
                self.transform(x_en_face),
                self.transform(x_rnfl_thickness),
                self.transform(x_gcl_thickness),
                self.transform(x_gcl_p)
            ]
            return x, y_label
        elif self.mode == 'ssl':
            raise Exception('Not implemented')

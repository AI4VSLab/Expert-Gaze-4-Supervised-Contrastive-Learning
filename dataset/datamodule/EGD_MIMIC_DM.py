from torch.utils.data import DataLoader
from dataset.utils.sampler import *
from torchvision import transforms
from dataset.transforms.custom_transforms import RandomCutOut, get_color_distortion
from dataset.dataset.EGD_MIMIC_dataset import EGD_MIMIC_Dataset
import glob
import pandas as pd
from collections import defaultdict
import torch
import numpy as np
import pytorch_lightning as pl
from nn.loss.supCon_loss import find_k_closest_reports, parition_cosine_similarity_eyeTrack
import sys
sys.path.append("/../../Self-Supervised-Learning/")


# transform for preprocess
transform_report_preprocess = transforms.Compose([
    transforms.Resize(
        (512, 512), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
])


# For whole report
transforms_report_simclr = transforms.Compose([
    transforms.RandomResizedCrop(
        size=(512, 512), scale=(0.4, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    # get_color_distortion(),
    RandomCutOut(size=(512, 512), min_cutout=0.1, max_cutout=0.7),
    transforms.RandomRotation(degrees=(0, 360)),
    # transforms.ToTensor()
])


class EGD_MIMIC_DM(pl.LightningDataModule):
    def __init__(self,
                 base_path: str = '',
                 data_type: str = 'imgNgaze',
                 max_seq_length: int = 512,
                 batch_size: int = 32,
                 split: float = 0.8,
                 split_trainset: float = 1.0,
                 mode: str = 'ssl',
                 embeddings_path: str = '',
                 linear_eval: bool = False,
                 r2n_path: str = '',
                 **kwargs
                 ):
        '''
        We iterate through all the images that we have, in case there are images that are in df_master and not the other
        way around

        @params:
        '''
        super().__init__()
        self.base_path = base_path
        self.max_seq_length = max_seq_length
        self.batch_size, self.split, self.split_trainset = batch_size, split, split_trainset
        self.mode = mode
        self.data_type = data_type
        self.linear_eval = linear_eval
        self.embeddings_path = embeddings_path
        self.r2n_path = r2n_path

        # =========================================== get paths  ===========================================
        self.get_paths()  # create the paths and lists
        train_idx, test_idx = self.get_sampled_split()  # sample idx
        self.get_split(train_idx, test_idx)  # idx to get train_idx

        self.save_hyperparameters(logger=False)

    #####################################################################################################################################
    #                                                  Setting Up Code
    #####################################################################################################################################

    def get_paths(self):
        self.df_master = pd.read_csv(f'{self.base_path}/master.csv')
        self.df_master.set_index('dicom_id', inplace=True)

        return

    def get_sampled_split(self):
        idx = torch.randperm(len(self.df_master))
        # take the first 80% for train
        train_idx = idx[:int(self.split*len(self.df_master))].tolist()
        test_idx = idx[int(self.split*len(self.df_master)):].tolist()  # take the other 20% for test

        # if split train
        if self.split_trainset < 1.0:
            idx = torch.randperm(len(train_idx))[:int(
                self.split_trainset*len(train_idx))]
            # idx[:int(self.split_trainset*len(self.train_idx))].tolist()
            train_idx = np.array(train_idx)[idx]
            print(f'Using {self.split_trainset} of trainset {len(train_idx)}')

        return train_idx, test_idx

    def get_split(self, train_idx, test_idx):
        # list if uniqueID/str for a scan
        self.train_imgID = np.array(self.df_master.index.to_list())[train_idx]
        self.test_imgID = np.array(self.df_master.index.to_list())[test_idx]
        return

    #####################################################################################################################################
    #                                                   Data Module Methods
    #####################################################################################################################################
    def setup(self,
              stage: str = ''
              ):
        if self.r2n_path:
            tmp = self.train_imgID.copy()
            self.load_embeddings(self.r2n_path)

        print('self.train_imgID', len(self.train_imgID))

        self.trainset = EGD_MIMIC_Dataset(
            imgID=self.train_imgID,
            df_master=self.df_master,
            data_type=self.data_type,
            max_seq_length=self.max_seq_length,
            transform=transform_report_preprocess if self.linear_eval else transforms_report_simclr,
            transform_ssl=transforms_report_simclr,
            mode=self.mode,
        )
        self.testset = EGD_MIMIC_Dataset(
            imgID=self.test_imgID,
            df_master=self.df_master,
            data_type=self.data_type,
            max_seq_length=self.max_seq_length,
            transform=transform_report_preprocess,
            transform_ssl=None,
            mode='test',
        )

    def train_dataloader(self, num_workers=32, batch_size=None, use_sampler=True):

        l = np.array(list(self.trainset.labels.values())
                     ) if use_sampler else []

        return DataLoader(
            self.trainset,
            batch_size=self.batch_size if not batch_size else batch_size,
            sampler=get_sampler(l),  # l
            num_workers=num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        return

    def state_dict(self):
        # called when we do a checkpoint, ckpt made by the trainer
        state = {
            'train_imgID': self.train_imgID,
            'test_imgID': self.test_imgID
        }
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.train_imgID, self.test_imgID = state_dict['train_imgID'], state_dict['test_imgID']
        return

    def load_embeddings(self, path):
        ftype = path.split('.')[-1]
        if ftype == 'npz':
            # assume their names are the following
            Mnpz = np.load(path, allow_pickle=True)
            self.r2n = Mnpz['r2n'].item()
            self.filename2label = Mnpz['filename2label'].item()
            self.features_train = Mnpz['features_train']
            self.features_test = Mnpz['features_test']
            self.train_imgID = Mnpz['train_imgID']
            self.test_imgID = Mnpz['test_imgID']

        elif ftype == 'ckpt':
            pass

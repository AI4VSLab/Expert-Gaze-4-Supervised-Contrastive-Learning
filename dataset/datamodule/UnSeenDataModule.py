from torch.utils.data import DataLoader
from dataset.utils.sampler import *
from torchvision import transforms
from dataset.transforms.custom_transforms import RandomCutOut, get_color_distortion
from dataset.dataset.report_dataset import Report_Simple_Dataset, Report_MultiModal_Dataset
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

transform_report_multimodal_preprocess = transforms.Compose([
    transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
])

# simclr augmentations
#    we always need this
transforms_report_resize = transforms.RandomResizedCrop(
    size=(512, 512), scale=(0.1, 1.0), antialias=True)

# For whole report
transforms_report_simclr = transforms.Compose([
    transforms.RandomResizedCrop(
        size=(512, 512), scale=(0.1, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    get_color_distortion(),
    RandomCutOut(size=(512, 512), min_cutout=0.1, max_cutout=0.7),
    transforms.RandomRotation(degrees=(0, 360)),

])

applier_oct = transforms.RandomApply(
    transforms=transforms_report_simclr, p=0.5)

# For different scans
transforms_report_multimodal = transforms.Compose([
    transforms.RandomResizedCrop(
        size=(224, 224), scale=(0.1, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    # get_color_distortion(),
    RandomCutOut(size=(224, 224), min_cutout=0.1, max_cutout=0.7),
    transforms.RandomRotation(degrees=(0, 360)),
])


region_box = {  # works well for this: RLS_038_OD_TC, each one (x, y)
    'circum_rnfl': [(0.058, 0.198), (0.537, 0.422)],
    'en_face': [(0.02, 0.6988), (0.2244, 0.24)],
    'rnfl_thickness': [(0.244, 0.7), (0.222, 0.238)],
    'rnfl_p': [(0.647, 0.22), (0.281, 0.4)],
    'gcl_thickness': [(0.6, 0.685), (0.16, 0.234)],
    'gcl_p': [(0.795, 0.688), (0.16, 0.234)]
}


class UnSeenDataModule(pl.LightningDataModule):
    def __init__(self,
                 multi_modal: bool = False,
                 base_path: str = '',
                 batch_size: int = 32,
                 split: float = 0.8,
                 split_trainset: float = 1.0,
                 mode: str = 'ssl',
                 linear_eval: bool = False,
                 embeddings_path: str = '',
                 **kwargs
                 ):
        '''
        We iterate through all the images that we have, in case there are images that are in df_labels and not the other
        way around

        @params:
        '''
        super().__init__()

        self.base_path = base_path
        self.batch_size, self.split, self.split_trainset = batch_size, split, split_trainset
        self.mode = mode
        self.linear_eval, self.multi_modal = linear_eval, False

        self.multi_modal = multi_modal

        # =========================================== get paths  ===========================================
        self.get_paths()  # create the paths and lists
        train_idx, test_idx = self.get_sampled_split()  # sample idx
        self.get_split(train_idx, test_idx)  # idx to get train_idx

        if embeddings_path:
            self.load_embeddings(embeddings_path)
        self.train_imgID_subsampled = []
        self.save_hyperparameters(logger=False)

    #####################################################################################################################################
    #                                                  Setting Up Code
    #####################################################################################################################################

    def get_paths(self):
        self.df_labels = pd.read_csv(f'{self.base_path}/labels.csv')

        self.imgID2path = {}
        self.imgID2label = {}

        imgPaths_accept = glob.glob(
            f'{self.base_path}/Acceptable_5k/*/Study Eye/*.png')
        imgPaths_unaccept = glob.glob(
            f'{self.base_path}/Unacceptable_2k/*/Study Eye/*.png')

        self.imgPaths = imgPaths_accept + imgPaths_unaccept
        self.labels = np.array([1.0 for _ in range(
            len(imgPaths_accept))] + [0.0 for _ in range(len(imgPaths_unaccept))])

        # not used since there are images not in df_label and df_label not in images

        def fn2unique(fn):
            fn = fn.split('/')[-1]
            unique_name = '_'.join(fn.split('_')[0:2])
            return unique_name

        for imgP, y in zip(self.imgPaths, self.labels):
            self.imgID2path[fn2unique(imgP)] = imgP
            self.imgID2label[fn2unique(imgP)] = y

        return

    def get_sampled_split(self):
        idx = torch.randperm(len(self.imgID2path))
        # take the first 80% for train
        train_idx = idx[:int(self.split*len(self.imgPaths))].tolist()
        # take the other 20% for test
        test_idx = idx[int(self.split*len(self.imgPaths)):].tolist()
        return train_idx, test_idx

    def get_split(self, train_idx, test_idx):
        # list if uniqueID/str for a scan
        self.train_imgID = np.array(list(self.imgID2path.keys()))[train_idx]
        self.test_imgID = np.array(list(self.imgID2path.keys()))[test_idx]

        return

    #####################################################################################################################################
    #                                                   Data Module Methods
    #####################################################################################################################################
    def setup(self,
              stage: str = '',
              # mode: str = 'ssl'
              ):

        idx = torch.randperm(len(self.train_imgID))
        idx_train = idx[:int(self.split_trainset *
                             len(self.train_imgID))].tolist()
        # self.train_imgID_subsampled would not hv nth if it was loaded back from ckpt
        if len(self.train_imgID_subsampled) == 0:
            self.train_imgID_subsampled = self.train_imgID
            if self.split_trainset != 1.0:
                self.train_imgID_subsampled = self.train_imgID[idx_train]

        if self.multi_modal:
            self.trainset = Report_MultiModal_Dataset(
                imgPaths=[self.imgID2path[i]
                          for i in self.train_imgID_subsampled],
                labels=np.array([self.imgID2label[i]
                                for i in self.train_imgID_subsampled]),
                transform=transform_report_preprocess if self.linear_eval else transforms_report_multimodal,
                transform_ssl=None,
                mode=self.mode,
            )
            self.testset = Report_MultiModal_Dataset(
                imgPaths=[self.imgID2path[i] for i in self.test_imgID],
                labels=np.array([self.imgID2label[i]
                                for i in self.test_imgID]),
                transform=transform_report_multimodal_preprocess,
                transform_ssl=None,
                mode='test',
            )

        else:
            self.trainset = Report_Simple_Dataset(
                imgPaths=[self.imgID2path[i]
                          for i in self.train_imgID_subsampled],
                labels=np.array([self.imgID2label[i]
                                for i in self.train_imgID_subsampled]),
                transform=transform_report_preprocess if self.linear_eval else transforms_report_simclr,
                transform_ssl=transforms_report_simclr,
                ret_rpt_name=True,
                mode=self.mode,
            )
            self.testset = Report_Simple_Dataset(
                imgPaths=[self.imgID2path[i] for i in self.test_imgID],
                labels=np.array([self.imgID2label[i]
                                for i in self.test_imgID]),
                transform=transform_report_preprocess,
                transform_ssl=None,
                mode='test',
            )

    def train_dataloader(self, num_workers=32, batch_size=-1, use_sampler=True):
        print(
            f'Creating Train Dataloader with {len(self.trainset)} and {len(self.train_imgID)} originally')
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size if batch_size == -1 else batch_size,
            sampler=get_sampler(self.trainset.labels.astype(
                int) if use_sampler else []),
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
            'test_imgID': self.test_imgID,
            'train_imgID_subsampled': self.train_imgID_subsampled
        }
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.train_imgID, self.test_imgID = state_dict['train_imgID'], state_dict['test_imgID']
        self.train_imgID_subsampled = state_dict['train_imgID_subsampled']
        return

    def load_embeddings(self, path):
        ftype = path.split('.')[-1]
        if ftype == 'npz':
            # assume their names are the following
            Mnpz = np.load(path, allow_pickle=True)

            # assume their names are the following
            Mnpz = np.load(path, allow_pickle=True)
            self.features_train = Mnpz['features_train']
            self.features_test = Mnpz['features_test']
            self.allCurrFixName_train = Mnpz['allCurrFixName_train']
            self.allCurrFixName_test = Mnpz['allCurrFixName_test']
            self.reports_train = Mnpz['reports_train']
            self.reports_test = Mnpz['reports_test']
            self.train_idx = Mnpz['train_idx']
            self.test_idx = Mnpz['test_idx']
            self.r2f_train = Mnpz['r2f_train']
            self.r2f_test = Mnpz['r2f_test']

        elif ftype == 'ckpt':
            pass

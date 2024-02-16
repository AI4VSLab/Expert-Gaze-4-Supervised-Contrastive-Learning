from torch.utils.data import DataLoader
from dataset.utils.sampler import *
from torchvision import transforms
from dataset.transforms.custom_transforms import RandomCutOut, get_color_distortion
from dataset.dataset.report_dataset import Report_Dataset, Report_EyeTracking_Dataset
import glob
import pandas as pd
from collections import defaultdict
import torch
import numpy as np
import pytorch_lightning as pl
from nn.loss.supCon_loss import find_k_closest_reports, parition_cosine_similarity_eyeTrack
import sys
sys.path.append("/../../Self-Supervised-Learning/")


# these are for reports
label_labelVarient = {
    'G': ['G', 'G_Suspects', 'G_Suspects_2'],
    'S': ['S', 'S_Suspects']
}

labelVarient_label = {
    'G': 'G',
    'G_Suspects': 'G',
    'G_Suspects_2': 'G',
    'S': 'S',
    'S_Suspects': 'S'
}

LABEL_TO_INDEX = {
    'G': 1,
    'S': 0
}

# last update 7.5.2023
expertise = {
    'EEM1': 'removed for anonymity',
    'EEM2': 'removed for anonymity',
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
    'EEM17': 'removed for anonymity',
    'EEM18': 'removed for anonymity',
    'EEM19': 'removed for anonymity',
    'EEM20': 'removed for anonymity'
}

EXPERTS = {'removed for anonymity',
           'removed for anonymity', 'removed for anonymity'}


# trasnfomr for preprocess
transform_report_preprocess = transforms.Compose([
    transforms.Resize(
        (512, 512), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
])

#  we always need this
transforms_report_resize = transforms.RandomResizedCrop(
    size=(512, 512), scale=(0.1, 1.0), antialias=True)


# simclr augmentations
transforms_report_simclr = transforms.Compose([
    transforms.RandomResizedCrop(
        size=(512, 512), scale=(0.4, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    # get_color_distortion(), # this hurts for our method since we have a very small dataset and report is very big
    RandomCutOut(size=(512, 512), min_cutout=0.1, max_cutout=0.7),
    transforms.RandomRotation(degrees=(0, 360)),

])


# trasnfomr legacy
transforms_report_simclr_legacy = [
    RandomCutOut(size=(512, 512), min_cutout=0.1, max_cutout=0.7),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomHorizontalFlip(p=0.5)
]

applier_oct = transforms.Compose([
    transforms_report_resize,
    transforms.RandomApply(transforms=transforms_report_simclr_legacy, p=0.5)
])


class ReportDataModule(pl.LightningDataModule):

    def __init__(self,
                 report_dataset_path: str = '',
                 fixation_path: str = '',
                 tobii_fixation_path: str = '',
                 report_fixation_path: str = '',
                 batch_size: int = 32,
                 split: float = 0.8,
                 split_trainset: float = 1.0,
                 mode: str = 'ssl',
                 exclude_test_list: list = [],
                 data_type: str = 'report',
                 embeddings_path: str = '',
                 testEntireSet: bool = False,
                 linear_eval: bool = False
                 ):
        '''
        @params:
            report_path: path to report, 
            report_fixation_path: for data_type == 'report_eyetrack_combined' else empty
            mode: 'ssl', 'sl' or 'test'
            exclude_test_list: list of imgs we are going to ignore
            data_type: 'report', 'report_eyetrack' , 'report_eyetrack_combined', 'report_eyetrack_puesdo'
            embeddings_path: for supcon
            split_trainset: how much of training data to use for training, between 0 to 1.0
        '''
        super().__init__()
        self.report_dataset_path, self.fixation_path, self.data_type, self.report_fixation_path = report_dataset_path, fixation_path, data_type, report_fixation_path
        self.batch_size = batch_size
        self.split, self.split_trainset = split, split_trainset
        self.tobii_fixation_path = tobii_fixation_path

        assert 0.0 <= self.split_trainset <= 1.0

        self.linear_eval = linear_eval

        # also mode
        self.mode = mode
        self.exclude_test_list = exclude_test_list
        self.embeddings_path = embeddings_path
        self.testEntireSet = testEntireSet

        self.reports_train_subsampled = []

        # make sure if we are using 'report_eyetrack', we have fixation path given
        if data_type == 'report_eyetrack':
            assert fixation_path
        if data_type == 'report_eyetrack_combined':
            assert report_fixation_path

        # =========================================== get paths  ===========================================
        self.get_vars()
        self.get_paths()  # create the paths and lists
        idx = self.get_sampled_split()  # sample idx
        self.get_split(idx)  # idx to get train_idx

        self.save_hyperparameters(logger=False)

    #####################################################################################################################################
    #                                                  Setting Up Code
    #####################################################################################################################################
    def get_vars(self):
        # palce holder for other data structures for getting paths, so we know what we have
        self.report_pavloviaFixation = self.report_label = self.report_path = self.reports = self.labels = self.train_idx = self.test_idx = None

    def get_paths(self):
        # =============================================== get paths for reports alone; label is for report ===============================================
        df_report, self.report_path, self.report_label = ReportDataModule.get_report_paths(
            self.report_dataset_path)
        self.reports, self.labels = np.array([k for k in self.report_path]), np.array(
            [1 if self.report_label[k] == 'G' else 0 for k in self.report_path])
        self.idx_reports, self.idx_reports = {r: i for i, r in enumerate(
            self.reports)}, {i: r for i, r in enumerate(self.reports)}

        # =============================================== get paths for eyetracking ===============================================
        if self.data_type == 'report_eyetrack_combined' or self.data_type == 'report_eyetrack_puesdo':
            self.report_pavloviaFixation, self.pavloviaFixation_report = ReportDataModule.get_pavlovia_fixation_paths(
                self.fixation_path)
            self.report_tobiiFixation, self.tobiiFixation_report = ReportDataModule.get_tobii_fixation_paths(
                self.tobii_fixation_path)
            # combine the two dicts
            self.report_fixation = defaultdict(list)
            for k in self.report_pavloviaFixation:
                self.report_fixation[k].extend(self.report_pavloviaFixation[k])
            for k in self.report_tobiiFixation:
                self.report_fixation[k].extend(self.report_tobiiFixation[k])

    def get_sampled_split(self):
        idx = torch.randperm(len(self.reports))
        return idx

    def get_split(self, idx):
        if self.data_type == 'report':
            # take the first 80% for train
            self.train_idx = idx[:int(self.split*len(self.reports))].tolist()
            # take the other 20% for test
            self.test_idx = idx[int(self.split*len(self.reports)):].tolist()
            self.reports_train, self.report_test = self.reports[
                self.train_idx], self.reports[self.test_idx]

        elif self.data_type == 'report_eyetrack_combined':
            # note really train_idx, but list of report names instead, so we dont hv to have differnet naems for saveckpt
            self.reports_train, self.report_test = ReportDataModule.build_train_test(
                self.report_fixation, self.split)

        elif self.data_type == 'report_eyetrack_puesdo':
            # load from here instead
            if self.embeddings_path:
                # load from embedding
                self.load_embeddings(self.embeddings_path)
                # get mask, NOTE: mask[i] corrspound to .allCurrFixName_train[i] ~ reports_train[i]
                self.r2n = find_k_closest_reports(
                    self.features_train,
                    self.allCurrFixName_train,
                    self.tobiiFixation_report,
                    self.pavloviaFixation_report)

    
                assert len(self.r2n) != 0
            # we would read the split from
            else:
                raise Exception(
                    'report_eyetrack_pseudo can only be used given embedding file')


    @staticmethod
    def get_tobii_fixation_paths(tobii_fixation_path=None):
        '''
        build dict[rpts] = [fixations]
        @params:
            tobii_fixation_path: path to processed tobii folder
        '''
        report_tobiiFixation = defaultdict(list)
        exp_folder = glob.glob(f'{tobii_fixation_path}/*/')
        for exp_f in exp_folder:
            # get name, ie EEM20
            exp_name = exp_f.split('/')[-2]
            exp_fix = glob.glob(f'{exp_f}*.csv')  # get list of csvs
            for fix in exp_fix:
                if fix.removesuffix('.csv').split('/')[-1] == exp_name:
                    continue
                _rpt = fix.removesuffix('.csv').split('/')[-1]  
                rpt = '_'.join(_rpt.split('_')[1:])  
                report_tobiiFixation[rpt].append(f'{exp_name}_{_rpt}')
        # the other way around
        tobiiFixation_report = defaultdict(str)

        for report in report_tobiiFixation:
            for fix in report_tobiiFixation[report]:
                tobiiFixation_report[fix] = report

        return report_tobiiFixation, tobiiFixation_report

    @staticmethod
    def get_pavlovia_fixation_paths(pavlovia_fixation_path=None):
        '''
        Used by data_type 'report_eyetrack'

        maps report names to fixation files 
        ie 'removed for anonymity ': ['removed for anonymity']

        @param:
            pavlovia_fixation_path: path to folder of fixation files
                                    NOTE: expect this data to be processed, with pavolovia.csv inside
        '''
        report_pavloviaFixation = defaultdict(list)
        df_pavlovia_fixation_data = pd.read_csv(
            f'{pavlovia_fixation_path}pavolovia.csv')
        df_pavlovia_fixation_data = df_pavlovia_fixation_data.dropna()
        df_pavlovia_fixation_data.set_index(
            "Processed_Data_Filename", inplace=True)

        for idx, row in df_pavlovia_fixation_data.iterrows():
            # exp_name = row.Experiment_Name
            # if expertise[exp_name.split('_')[0]] != 'removed for anonymity' and expertise[exp_name.split('_')[0]] != 'removed for anonymity' and expertise[exp_name.split('_')[0]] != 'removed for anonymity': continue
            report_pavloviaFixation[row.Report_Used].append(idx)

        # the other way
        pavloviaFixation_report = defaultdict(str)
        for report in report_pavloviaFixation:
            for fix in report_pavloviaFixation[report]:
                pavloviaFixation_report[fix] = report

        return report_pavloviaFixation, pavloviaFixation_report

    @staticmethod
    def get_report_paths(report_base_path=None):
        '''
        skips exclude list
        @params:
            report_base_path: path to actual report images folder

        @returns
            - df_report: this one is for looking
            - report_path:  report_path[full path] = [list of paths]

            - report_label: report_labels[report name] = labels
                some reports might have different labels like 'S', 'S_Suspect' and 'G'. choose using majority voting
        '''
        # these two below are quite useless since they map path -> 'G' but same path appears more than once
        # since each report is used more than once
        fn = []  # stores full path
        labels = []

        # same as below except we choose a lable now
        report_label = defaultdict(str)
        # report_labels[report name] = ['G', 'G_suspects]
        report_labels = defaultdict(list)
        # report_paths[report name ] = [paths to report]
        report_paths = defaultdict(list)

        '''
        report_folder_all_paths = {
                            'G': f'{report_base_path}G/',
                            'S': f'{report_base_path}S/',
                            'G_Suspects': f'{report_base_path}G_Suspects/',
                            'G_Suspects_2': f'{report_base_path}G_Suspects_2/',
                            'S_Suspects': f'{report_base_path}S_Suspects/'
            }       
        '''
        report_folder_all_paths = glob.glob(f'{report_base_path}/*/')

        print('\n')
        for folderPath in report_folder_all_paths:
            currlabel = folderPath.split('/')[-2]

            curr_len = len(fn)
            paths = glob.glob(f"{folderPath}*.png", recursive=False) + \
                glob.glob(f"{folderPath}*.jpg", recursive=False)
            for p in paths:
                report_name = p.removeprefix(folderPath).split(
                    '.')[0]  # get rid of prefix and .jpg for ex
                report_paths[report_name].append(p)
                report_labels[report_name].append(currlabel)
                labels.append(currlabel)
                fn.append(report_name)

            print(f'{currlabel} has {len(fn)-curr_len}')
        print('\n')

        df_report = pd.DataFrame(list(zip(labels, fn)),
                                 columns=['Labels', 'Filename'])

        for report_p in report_labels:
            # get the name only, strip path and .png/.jpg for ex
            report_name = report_p.split('/')[-1].split('.')[0]
            num_G, num_S = 0, 0
            for l in report_labels[report_p]:
                if labelVarient_label[l] == 'G':
                    num_G += 1
                else:
                    num_S += 1
            # break even arbitary
            if num_G > num_S:
                report_label[report_name] = 'G'
            else:
                report_label[report_name] = 'S'

        return df_report, report_paths, report_label

    @staticmethod
    def build_train_test(report_pavloviaFixation, split=0.8):
        '''
        create disjoint train test sets based on actual report; first choose reports with>=5 fixations for train; then same from rest to 
        make up to 80%.

        @returns:
             trainset_reports: list with [(report 1, fixation 1 for report 1), (report 1, fixation 2 for report 1),...]
             testset_reports: same as above but for test
        '''
        # build train, we take ones with >5 to train
        num_fixations = 0
        num_reportsUsed_fixation = len(report_pavloviaFixation)
        for fixations in report_pavloviaFixation.values():
            num_fixations += len(fixations)

        print(
            f'There are {num_fixations} total number of fixations and {num_reportsUsed_fixation} reports used')
        # approach to building trainset: use all the fixations on reports with >= 5 fixations, which would be a portion of 80% of train for example
        # sample from the rest to make up to 80% if not enough
        train_reports, train_candidate = [], []  # we are using train_reports for sure

        for idx, report_name in enumerate(report_pavloviaFixation):
            if len(report_pavloviaFixation[report_name]) >= 5:
                for fixation_fn in report_pavloviaFixation[report_name]:
                    train_reports.append((report_name, fixation_fn))
            else:
                for fixation_fn in report_pavloviaFixation[report_name]:
                    train_candidate.append((report_name, fixation_fn))

        train_reports, train_candidate = np.array(
            train_reports), np.array(train_candidate)

        # total num of fixations in reports we are using for sure
        train_reports_numFixations = len(train_reports)
        # total num of fixations in candidate reports
        train_candidate_numFixations = len(train_candidate)

        # shuffle the indices for every data randomly and split into train and test
        split_makeup = (int(num_fixations*split) -
                        train_reports_numFixations)/train_candidate_numFixations
        print(f'We need {split_makeup}% examples from train_candidate')

        # These are idx into train_candidate, not report_pavloviaFixation
        idx = torch.randperm(len(train_candidate))
        # take the first 80% for train
        idx_train = idx[:int(split*len(train_candidate))].tolist()
        # take the other 20% for test
        idx_test = idx[int(split*len(train_candidate)):].tolist()

        # now putting everything tgt
        # [ [report name, fixation file name],...]
        trainset_reports = np.vstack(
            (train_reports, train_candidate[idx_train]))
        testset_reports = train_candidate[idx_test]

        print(
            f'There are {len(trainset_reports)} training reports and {len(testset_reports)} testing reports ')
        return trainset_reports, testset_reports

    @staticmethod
    def get_split_simple(split,
                         reports,
                         report_tobiiFixation,
                         report_pavloviaFixation,
                         reports_train=[],
                         reports_test=[],
                         gaze_data=None
                         ):
        '''
        called by eye tracking modules, sample 80% reports, and the number of fixations we get as a results is what will be 

        used to train gaze modesl
        '''
        idx = None
        print(f'we have this many reports {len(reports)}')
        train_idx = test_idx = []
        if len(reports_train) == 0 and len(reports_test) == 0:
            idx = torch.randperm(len(reports))
            # take the first 80% for train
            train_idx = idx[:int(split*len(reports))].tolist()
            # take the other 20% for test
            test_idx = idx[int(split*len(reports)):].tolist()
            reports_train = reports[train_idx]
            reports_test = reports[test_idx]
        # we would read the split from
        r2f_train, r2f_test = ReportDataModule.get_split_reports_fixation_pair(
            reports_train, reports_test,
            reports,
            report_tobiiFixation,
            report_pavloviaFixation,
            list(gaze_data.index)
        )
        return train_idx, test_idx, r2f_train, r2f_test, reports_train, reports_test

    @staticmethod
    def get_split_reports_fixation_pair(reports_train, reports_test, reports, report_tobiiFixation, report_pavloviaFixation, gaze_valid):
        ''' 
        train, test_idx for reports, build reports_train[rpt] = [fixation names] 
        NOTE: get rid of EEM 13 and only include the one that are valid from gaze
        '''
        r2f_train, r2f_test = defaultdict(list), defaultdict(list)
        # reports_train, reports_test = reports[train_idx], reports[test_idx]
        for r in reports_train:
            fixs = report_tobiiFixation[r] + report_pavloviaFixation[r]
            r2f_train[r].extend([f for f in fixs if (
                'EEM13' not in f and f in gaze_valid)])
        for r in reports_test:
            fixs = report_tobiiFixation[r] + report_pavloviaFixation[r]
            r2f_test[r].extend([f for f in fixs if (
                'EEM13' not in f and f in gaze_valid)])
        return r2f_train, r2f_test

    #####################################################################################################################################
    #                                                   Data Module Methods
    #####################################################################################################################################
    def setup(self,
              stage: str = '',
              # mode: str = 'ssl'
              ):
        '''
        instantie pytorch datasets and train/val/test split
        '''
        print(f'Setting up for {stage} now with mode {self.mode}')

        # we call setup after load_from_checkpoint as well, but it will reset self.reports to [] for some reason hence call them again
        if len(self.reports) == 0:
            self.get_paths()
            print(f'len {len(self.reports)}')

        if self.data_type == 'report':
            self.trainset = Report_Dataset(
                reports=self.reports_train_subsampled,
                report_label=self.report_label,
                report_path=self.report_path,
                transform=transform_report_preprocess if self.linear_eval else transforms_report_simclr,
                transform_ssl=transforms_report_simclr,  # transforms_report_simclr,
                mode=self.mode,
            )

            self.testset = Report_Dataset(
                reports=self.reports[self.test_idx] if not self.testEntireSet else self.reports,
                report_label=self.report_label,
                report_path=self.report_path,
                transform=transform_report_preprocess,
                transform_ssl=None,
                mode='test',
            )
        elif self.data_type == 'report_eyetrack_combined':
            self.trainset = Report_EyeTracking_Dataset(
                reports=self.reports_train,  # these are actually reports
                pavlovia_combined_path=self.report_fixation_path,
                report_pavloviaFixation=self.report_pavloviaFixation,
                report_label=self.report_label,
                report_path=self.report_path,
                transform=transform_report_preprocess if self.linear_eval else self.transforms_report_simclr,
                transform_ssl=transforms_report_simclr,
                mode=self.mode,
            )
            self.testset = Report_EyeTracking_Dataset(
                reports=self.reports_test,  # these are actually reports
                pavlovia_combined_path=self.report_fixation_path,
                report_pavloviaFixation=self.report_pavloviaFixation,
                report_label=self.report_label,
                report_path=self.report_path,
                transform=transform_report_preprocess,
                mode='test',
            )
        elif self.data_type == 'report_eyetrack_puesdo':
            

            # sub sample a new set
            if len(self.reports_train_subsampled) == 0 and self.split_trainset < 1.0:
                idx = torch.randperm(len(self.reports_train))
                idx_train = idx[:int(
                    self.split_trainset*len(self.reports_train))].tolist()
                self.reports_train_subsampled = self.reports_train[idx_train]
            elif len(self.reports_train_subsampled) > 0 and self.split_trainset < 1.0:
                # further subsample
                idx = torch.randperm(len(self.reports_train_subsampled))
                idx_train = idx[:int(
                    self.split_trainset*len(self.reports_train_subsampled))].tolist()
                self.reports_train_subsampled = self.reports_train_subsampled[idx_train]
            else:
                self.reports_train_subsampled = self.reports_train


            self.trainset = Report_Dataset(
                reports=self.reports_train_subsampled,
                report_label=self.report_label,
                report_path=self.report_path,
                ret_rpt_name=True,
                transform=transforms_report_resize,  # transform_report_preprocess,
                transform_ssl=transforms_report_simclr,
                mode=self.mode,
            )

            self.testset = Report_Dataset(
                reports=self.reports_test,
                report_label=self.report_label,
                report_path=self.report_path,
                transform=transform_report_preprocess,
                transform_ssl=None,
                mode='test',
            )

    def train_dataloader(self, num_workers=32, batch_size=None, use_sampler=True):
        print(
            f'Creating a new trainloader {self.trainset.mode} with {len(self.trainset)}')
        # self.train_idx for datasets with fixation arent actual index, so need diff ways to get label
        if self.data_type == 'report':
            l = []
            if self.mode == 'sl':
                l = [int(LABEL_TO_INDEX[self.report_label[r]])
                     for r in self.trainset.reports]
        elif self.data_type == 'report_eyetrack_puesdo':
            l = []
            # use self.trainset.reports since it might take a smaller set
            if self.mode == 'sl':
                l = [int(LABEL_TO_INDEX[self.report_label[r]])
                     for r in self.trainset.reports]  # self.labels[self.train_idx]
        else:
            l = np.array([1 if self.report_label[rpt] ==
                         'G' else 0 for (rpt, fix) in self.train_idx])
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size if not batch_size else batch_size,
            sampler=get_sampler(l if use_sampler else []),
            num_workers=num_workers
        )

    def val_dataloader(self):
        # use testset for val for now
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
            'reports_train': self.reports[self.train_idx],
            'reports_test': self.reports[self.test_idx],
            'reports_train_subsampled': self.reports_train_subsampled
        }
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        # self.train_idx, self.test_idx = state_dict['train_idx'], state_dict['test_idx']
        self.reports_train, self.reports_test = state_dict['reports_train'], state_dict['reports_test']
        try:
            self.reports_train_subsampled = state_dict['reports_train_subsampled']
        except:
            print('self.reports_train_subsampled not in ckpt, maybe old version')

    def load_embeddings(self, path):
        ftype = path.split('.')[-1]
        if ftype == 'npz':
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

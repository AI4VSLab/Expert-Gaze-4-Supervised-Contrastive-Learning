from typing import Any
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import torch

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


def _get_train_test_split(
    pavlovia_data_path,
    df_pavlovia_data,
    split=0.8
):
    folders_path = glob.glob(f"{pavlovia_data_path}*/")

    # store data on list
    # X[i].shape = 256 right now
    X = []
    Y = []
    X_filename = {}
    expLevel = []

    for folder in folders_path:
        # only if start of folder start with EEM
        if folder.split('/')[-2][0:3] != 'EEM':
            continue

        # get path to npz file
        # there should only be one file
        npz_path = glob.glob(f'{folder}*.npz')[0]
        imgs_loaded = np.load(npz_path)

        for file in imgs_loaded.files:
            exp_num = file.split('_')[0]  # this is EEM13 for ex

            try:
                if exp_num not in expertise:
                    continue
                expLevel.append(1.0 if (expertise[exp_num] == 'removed for anonymity' or expertise[exp_num] ==
                                'removed for anonymity' or expertise[exp_num] == 'removed for anonymity') else 0.0)  # lets use 1 for expert 0 not expert
            except:
                print('we dont have expertise yet?')

            X.append(imgs_loaded[file].flatten())
            Y.append(
                1 if df_pavlovia_data.loc[file].Label == 'Glaucoma' else 0)
            X_filename[file] = len(X)-1

    print(
        f'There are {np.sum(expLevel)} number of experts and {len(expLevel)-np.sum(expLevel)} non-experts')
    print(
        f'There are {np.sum(Y)} number of glaucoma and {len(Y)-np.sum(Y)} healthy')

    # ================== get split ==================
    X, Y, expLevel = torch.tensor(X), torch.tensor(Y), torch.tensor(expLevel)

    idx = torch.randperm(len(X))
    # take the first 80% for train
    train_idx = idx[:int(split*len(X))].tolist()
    test_idx = idx[int(split*len(X)):].tolist()  # take the other 20% for test

    X_train, Y_train, expLevel_train = X[train_idx], Y[train_idx], expLevel[train_idx]
    X_test, Y_test, expLevel_test = X[test_idx], Y[test_idx], expLevel[test_idx]

    return X_train, Y_train, expLevel_train, X_test, Y_test, expLevel_test,  X, X_filename


class Pavlovia_EyeTracking(Dataset):
    def __init__(self,
                 pavlovia_data_path='',
                 transform=None,
                 mode='sl',
                 get_expertise=False,
                 X=None,
                 Y=None,
                 expLevel=None
                 ) -> None:
        '''
        dataset has differnet modes, if 'self-supervised-simclr' then we will return two verison of the same image 

        also can get expert level if True. But some experiments are not included if not an expert (EEM2 for example)
        '''
        super().__init__()
        self.transform = transform
        self.mode = mode
        self.get_expertise = get_expertise

        # read main df
        self.df_pavlovia_data = pd.read_csv(
            f'{pavlovia_data_path}pavolovia.csv')
        self.df_pavlovia_data = self.df_pavlovia_data.dropna()
        self.df_pavlovia_data.set_index(
            "Processed_Data_Filename", inplace=True)

        self.X, self.Y, self.expLevel = torch.tensor(
            X), torch.tensor(Y), torch.tensor(expLevel)

    def configure_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: Any) -> Any:
        x = self.X[idx].view(16, 16).unsqueeze(0)
        if self.mode == 'sl':
            if self.transform:
                x = self.transform(x)
            if self.get_expertise:
                return x, self.Y[idx], self.expLevel[idx]
            return x, self.Y[idx]

        elif self.mode == 'ssl':
            # assume we have self.transform  since we need it
            # also assume transfomr is random transform so gives differnet transforms
            x_i, x_j = self.transform(x), self.transform(x)

            if self.get_expertise:
                return x_i, x_j, self.Y[idx], self.expLevel[idx]

            return x_i, x_j, self.Y[idx]

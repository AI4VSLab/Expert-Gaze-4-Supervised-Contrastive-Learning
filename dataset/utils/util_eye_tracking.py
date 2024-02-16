'''
util for data already in eye-tracking data folder. we have separate code for pavolvia/tobii -> eye-tracking data folder

code here are used for merging csvs for example
'''
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataset.datamodule.ReportDataModule import expertise
import sys
sys.path.append("/../../Self-Supervised-Learning/")


########################################################################################################
#                                      dataset construction util
########################################################################################################
def combine_label(ppath, tpath, target_folder):
    '''
    using current processing pipeline, which uses pavolovia.csv made by code in this repo and tobii proessing pipeline
    each one of them would have a csv file that contains labels. This function is used to combine them

    NOTE: these are actual labels, not what participants rate when they see the image
    @params:
        ppath: path to ../pavolivia.csv
        tpath: path to ../label.csv in tobii
    '''
    dfp, dft = pd.read_csv(ppath), pd.read_csv(tpath).dropna()

    d = defaultdict(str)

    # build dict map rpt name to laabel
    for idx, row in dfp.iterrows():
        # make sure they all have the same label
        if d[row.Report_Used]:
            assert d[row.Report_Used] == row.Label
        d[row.Report_Used] = row.Label
    for idx, row in dft.iterrows():
        if d[row.report]:
            assert d[row.report] == row.label
        d[row.report] = row.label

    df_combined = pd.DataFrame(
        list(zip(list(d.keys()), list(d.values()))), columns=['report', 'label'])

    df_combined.to_csv(f'{target_folder}/label.csv')


########################################################################################################
#                                      Visualization
########################################################################################################
def histogram(path, title='', only_expert=True, df_gaze_data=None, exclude_set={}, dpi=1200, figsize=(8, 8)):

    items = os.listdir(path)
    folders = [item for item in items if os.path.isdir(
        os.path.join(path, item))]

    data_lengths = []
    data_in_array = []
    csv_used = []
    count_nan = 0

    for folder in folders:  # folder ie EEM18 or EEM1_fixations
        if only_expert:
            if not (expertise[folder.split('_')[0]] == 'removed for anonymity' or expertise[folder.split('_')[0]] == 'removed for anonymity' or expertise[folder.split('_')[0]] == 'removed for anonymity'):
                continue
        # '/../../data/eye_tracking_data/EEM11_fixations'
        folder_path = os.path.join(path, folder)

        csvs = os.listdir(folder_path)

        for csv in csvs:
            # there are also gaze that we skip
            # csv.removesuffix('.csv') not  in df_gaze_data.index  or
            if csv.removesuffix('.csv') in exclude_set or csv.removesuffix('.csv') not in df_gaze_data.index:
                continue

            csv_path = os.path.join(folder_path, csv)
            df = pd.read_csv(csv_path)  # .dropna() # na in data denotes tho
            data_lengths.append(len(df))
            csv_used.append(csv.removesuffix('.csv'))

    print('number of nans: ', count_nan, 'max length', max(data_lengths))

    plt.figure(dpi=dpi, figsize=figsize)
    # You can adjust the number of bins and colors as needed
    plt.hist(data_lengths, bins=200, color='blue', edgecolor='black')
    plt.xlabel('Number of Fixations')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

    return sorted(data_lengths)

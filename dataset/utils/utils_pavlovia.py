'''util for all pavlovia data stuff'''

from dataset.utils.gaze_filtering import *
from torchvision.io import read_image


import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
import torch

import pandas as pd
import glob
import sys
import os
import cv2


def show(imgs):
    '''function from: https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py'''
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15, 15))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def one_hot(x, N):
    '''convert x which int that represents class to one hot vector'''
    one_hot = torch.zeros(N)
    one_hot[x.long()] = 1.0
    return one_hot


#####################################################################################################################################
#                                       Building Pavlovia Fixation Dataset and Store
#####################################################################################################################################
def pavlovia_clean_csv(fpath):
    '''change column names '''
    df = pd.read_csv(fpath)

    # drop columns not used used; NOTE: we dont need current timestamp bc we can just used duration
    df = df.drop(columns=['world_index', 'fixation_id', 'world_timestamp',
                 'start_timestamp', 'on_surf', 'x_scaled', 'y_scaled', 'dispersion'])

    # change name of columns
    df = df.rename(columns={"norm_pos_x": "x_norm", "norm_pos_y": "y_norm"})

    return df


def pavlovia_fixations_to_images(fpath, img_x_size=16, img_y_size=16):
    '''
    convert fixation to an image, each coordinate is weight by how long it was looked at 
    @params
      - fpath: path to csv for a given scan, ie ../fixations_Glaucoma_XXXXXXXX_widefield_report.csv

    @returns
      - img: processed image
      - flag: -1 if df is empty -> faulty calibration and more hence no data, we will skip them
    '''

    df = pd.read_csv(fpath)

    img = np.zeros((img_x_size, img_y_size), float)
    if len(df) == 0:
        return img, -1

    coord = []

    for idx, row in df.iterrows():
        x_coord, y_coord = int(
            row.norm_pos_x*img_x_size), int(row.norm_pos_y*img_y_size)
        coord.append((x_coord, y_coord))
        img[x_coord][y_coord] += 1*row.duration  # weighted by duration

    # make sure to catch the ones that recorded nothing
    with np.errstate(all='raise'):
        try:
            img /= np.max(img)
        except:
            print(fpath)
    return img, 0


def pavlovia_read_feedback(pavlovia_path, save_path=''):
    from collections import defaultdict
    from dataset.datamodule.EyeTrackingDM import EyeTrackingDM

    path_csv = glob.glob(f'{pavlovia_path}/*.csv')
    sub_dir = glob.glob(f'{pavlovia_path}/*/')

    # fiter fixaiton files and pavolovia.csv which contains dataset summary
    path_csv = [p for p in path_csv if (p.removesuffix('.csv').split(
        '_')[-1] != 'fixations' and p.removesuffix('.csv').split('/')[-1] != 'pavolovia')]

    pavolovia_df = pd.read_csv(f'{pavlovia_path}pavolovia.csv')

    # ============================================= read num files in each folder =============================================
    # subDir_numFiles[EEM9_control] = number of fixations in folder
    subDir_numFiles = defaultdict(int)
    subDir_Files = defaultdict(list)
    for path_sub_dir in sub_dir:
        paths_in_subDir = glob.glob(f'{path_sub_dir}*.csv')

        tmpName = path_sub_dir.split('/')[-2]  # EEM11_fixations for ex
        currExpName = '_'.join(tmpName.split('_')[:-1])
        subDir_numFiles[currExpName] = len(paths_in_subDir)
        subDir_Files[currExpName] = paths_in_subDir

    # ============================================= read dicts =============================================
    report_pavloviaFixation, pavloviaFixation_report = EyeTrackingDM.get_reportFix_map(
        pavlovia_path, tobii_path=None, getPavOnly=True)

    exp_name = list(pavolovia_df.Experiment_Name)
    # ============================================= read comments =============================================
    # create dictionary dict[EEM4_XXXXXXXX_widefield_report] = expert feedback
    currFixName_score = {}
    currFixName_comment = {}
    currFixName_notValid = {}  # store the ones that are not valid

    FixName = []
    score = []
    comment = []

    for expMetaPath in path_csv:
        df_exp = pd.read_csv(expMetaPath)
        currExpName = '_'.join(expMetaPath.split(
            '/')[-1].split('_')[:-1])  # for example EEM6_2

        # drop na, but only if no files shown; len(df_exp) should be the same as number of fixations in an experiment
        df_exp = df_exp.dropna(subset=['img_filename'])
        if len(df_exp) != subDir_numFiles[currExpName]:
            print(
                f'There are more responses than parsed fixations {expMetaPath}')

        for i, row in df_exp.iterrows():
            # print(row.img_filename, row['response_text.text'],row['OCTresponse_slider.response'])
            currImgName = row.img_filename.split('.')[0]
            currFixName = f'{currExpName}_{ currImgName}'

            # put in dict
            try:
                currFixName_score[currFixName] = int(
                    row['OCTresponse_slider.response'])
                currFixName_comment[currFixName] = str(
                    row['response_text.text'])
            except:
                currFixName_notValid[currFixName] = True
                print(f'This file has Nan response {expMetaPath}')
                print(f'looking at {currFixName}',
                      row['OCTresponse_slider.response'], row['response_text.text'])
                print()

            # put in list
            FixName.append(currFixName)
            score.append(row['OCTresponse_slider.response'])
            comment.append(row['response_text.text'])

    df_feedback = pd.DataFrame(
        {'FixName': FixName,
         'score': score,
         'comment': comment
         })
    df_feedback.__description__ = 'Nan means participants did not input values'
    if save_path:
        df_feedback.to_csv(save_path)

    # ============================================= look at the ones missed =============================================
    print('This are not used becasue errors with calibration or other reasons')
    for n in FixName:
        if n not in pavloviaFixation_report:
            print(n)
    return FixName, score, comment


def pavlovia_build_dataset(
    base_path,
    oct_repaired_dict,
    report2subfolder,
    get_region_box,  # function to get region box from oct_repaired_dict
    report2path,
    img2flip,
    img_size=(16, 16),
    save_mode='batch',
    save_data='fixImg',
    target_folder=None,
    time_bin=5,
    box_size=None,
    box_grid_size={},
    report_path=None,
    word_size=(0.05, 0.1)
):
    '''
    convert fixation to images, and save all fixations in all experiments (ie EEM1_fixation) in basefolder
    to a npy file inside EEM1_fixation/ folder for example
    also create df to store metadata

    This only works with current directory setup ie 

    base_path
      -> EEM1_fixations
        -> all the csvs

    @params: 
      base_path: path to main data folder 
      save_mode: 'batch' save everything in current folder ie EEM1 into a single npzfile 
                 'single' save each one into a pytorch .pt file, same name 
      save_data: what data we want to save. 'fixImg' and 'time_series'; NOTE: each save_data would return a slightly different csv and have 
        different target_folder. Almost behaving like a different function
          For the different data: 
          'time_series': we should use save_mode == 'single' for saving to combined folder with tobiil; img_size is used as how to split image into grid
      target_folder: used with save_mode single
      time_bin: [ms], how long each bin is 
      report_path: dict report_path[rpt name] = path to report
      img_size: 
        when save_mode == time_serie, ex {
                                                  'view':(16,32), 
                                                  'RLS': (16,25)
                                          }
    '''
    folders_path = glob.glob(f"{base_path}/*/")

    # create an image for each corrspound report, stored in the experiment's folder
    # also create a df to store info
    df_list = []  # we will append rows
    rls_list = []
    view_list = []

    for folder in folders_path:
        csv_paths = glob.glob(f"{folder}/*.csv")
        # EEM15_control_fixations for example -> now just EEM15_control
        experiment_name = '_'.join(
            folder.removeprefix(base_path)[:-1].split('_')[:-1])

        # make dir if dir not exist
        if not os.path.isdir(f'{target_folder}/{experiment_name}'):
            os.mkdir(f'{target_folder}/{experiment_name}')

        imgs_processed = {}
        for csv_file in csv_paths:
            # fixations_Glaucoma_8914_OD_2021_widefield_report for example
            csv_file_filename = csv_file.removeprefix(folder).split('.')[0]

            label, report_used = csv_file_filename.split(
                '_')[1], '_'.join(csv_file_filename.split('_')[2:])

            csv_file_filename_new = '_'.join([experiment_name, report_used])

            # ========================================= read region box =========================================
            box_size = get_region_box(
                report_used, report2subfolder[report_used], oct_repaired_dict, report2path)

            # ========================================= turn fixation into an image =========================================
            if save_data == 'fixImg':
                img, img_flag = pavlovia_fixations_to_images(
                    csv_file, img_size[0], img_size[1])
                if img_flag != -1:
                    imgs_processed[csv_file_filename_new] = img
                    if save_mode == 'single':
                        torch.save(
                            img, f'{target_folder}/{csv_file_filename_new}')

                # append to df_list for current row, need to store the experiment name as well since same image would be used in different exp
                csv_file_filename = csv_file_filename_new if img_flag != -1 else np.nan

            # ========================================= clean csv and bin them based on time =========================================
            if save_data == 'time_series':
                # first make a new df based on current
                df_cleaned = pavlovia_clean_csv(csv_file)
                # NOTE: skips empty df
                if len(df_cleaned) == 0:
                    continue

                na_filter = Nan_Filter()
                df_cleaned = na_filter(df_cleaned)
                sample_filter = Sampling_Filter(time_bin)
                fix_cleaned_filtered = sample_filter(df_cleaned)
                df_cleaned_filtered = pd.DataFrame(
                    fix_cleaned_filtered, columns=['x_norm', 'y_norm'])

                quan_filter = Uniform_Quantize_Filter(img_size)
                df_quan_filter = quan_filter(df_cleaned_filtered, report_used)

                reg_filter = Region_Filter(
                    box_size, word_size, box_grid_size, img2flip if 'OD' in csv_file_filename_new else {})
                df_reg_filter = reg_filter(df_quan_filter)

                # NOTE add this back
                df_reg_filter.to_csv(
                    f'{target_folder}/{experiment_name}/{csv_file_filename_new}.csv')

            df_list.append([experiment_name, report_used,
                           csv_file_filename_new, label])

        if save_mode == 'batch':
            np.savez(f'{folder}processedImgs', **imgs_processed)
    print(rls_list)
    print(view_list)
    df = pd.DataFrame(df_list, columns=[
                      'Experiment_Name', 'Report_Used', 'Processed_Data_Filename', 'Label'])

    return df

'''util for all tobii data stuff'''
import glob
import pandas as pd
from dataset.utils.gaze_filtering import *
import os


def tobii_clean_csv(fpath):
    df = pd.read_csv(fpath,  index_col=0)  # no unnamed index column

    df = df.drop(columns=['report', 'time start [ms]', 'time end [ms]'])
    df = df.rename(columns={
                   'duration [ms]': "duration", 'fixation point x [mcs norm]': "x_norm", 'fixation point y [mcs norm]': "y_norm"})

    return df


def tobii_read_feedback(fpath):
    '''
    Build list that reads expert feedback
    '''
    FixName = []
    score = []
    comment = []

    sub_dir = glob.glob(f'{fpath}/*/')
    for subDirPath in sub_dir:
        currExpName = subDirPath.split('/')[-2]
        df_currExpMeta = pd.read_csv(f'{subDirPath}{currExpName}.csv')
        currFixName_all = [
            f'{currExpName}_{r}' for r in list(df_currExpMeta.report)]
        FixName.extend(currFixName_all)
        score.extend(df_currExpMeta.score)
        comment.extend(df_currExpMeta.comments)
    return FixName, score, comment


def tobii_build_dataset(
    base_path,
    oct_repaired_dict, 
    report2subfolder, 
    get_region_box,  # function to get region box from oct_repaired_dict
    report2path, 
    img2flip,
    img_size=(16, 16),
    target_folder=None,
    time_bin=5,
    box_size=None,
    box_grid_size={},
    report_path=None,
    word_size=(0.05, 0.1)
):

    exp_folder = glob.glob(f'{base_path}/*/')
    for exp_f in exp_folder:
        # get name, ie EEM20
        exp_name = exp_f.split('/')[-2]
        exp_fix = glob.glob(f'{exp_f}*.csv')  # get list of csvs

        # make dir if dir not exist
        if not os.path.isdir(f'{target_folder}/{exp_name}'):
            os.mkdir(f'{target_folder}/{exp_name}')

        for fix in exp_fix:
            n = fix.removesuffix('.csv').split('/')[-1]  # g_RLS... for ex
            if n == exp_name:
                continue  # skip EEM20.csv for ex
            report_used = n[2:]
            # ========================================= read region box =========================================
            box_size = get_region_box(
                report_used, report2subfolder[report_used], oct_repaired_dict, report2path)

            df_cleaned = tobii_clean_csv(fix)
            na_filter = Nan_Filter()
            df_cleaned = na_filter(df_cleaned)
            sample_filter = Sampling_Filter(time_bin)
            fix_cleaned_filtered = sample_filter(df_cleaned)

            df_cleaned_filtered = pd.DataFrame(
                fix_cleaned_filtered, columns=['x_norm', 'y_norm'])

            quan_filter = Uniform_Quantize_Filter(img_size)
            df_quan_filter = quan_filter(df_cleaned_filtered, n[2:])

            reg_filter = Region_Filter(
                box_size, word_size, box_grid_size, img2flip if 'OD' in report_used else {})
            df_reg_filter = reg_filter(df_quan_filter)

            df_reg_filter.to_csv(
                f'{target_folder}/{exp_name}/{exp_name}_{n}.csv')

    return

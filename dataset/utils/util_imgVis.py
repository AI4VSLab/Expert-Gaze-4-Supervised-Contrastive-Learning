'''util to show images'''
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataset.utils.util_read import read_img
import sys
sys.path.append("/../../Self-Supervised-Learning/")


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_eye_track_scatter(img_path, fix_path, source='pavlovia', show_plt=True, save_path=''):
    img = read_img(img_path)  # RLS_038_OD_TC 9261_OS_2021_widefield_report
    num_channels, y_size, x_size = img.shape

    df = pd.read_csv(fix_path)
    coord_x = []
    coord_y = []
    for idx, row in df.iterrows():
        if source == 'pavlovia':
            x_coord, y_coord = int(
                row.norm_pos_x*x_size), int(row.norm_pos_y*y_size)
        elif source == 'tobii':
            x_coord, y_coord = int(row['fixation point x [mcs norm]']
                                   * x_size), int(row['fixation point y [mcs norm]']*y_size)
        elif source == 'standard':
            x_coord, y_coord = int(
                row['x_norm']*x_size), int(row['y_norm']*y_size)
        coord_x.append(x_coord)
        coord_y.append(y_coord)

    fig = plt.figure(figsize=(16, 8))
    plt.imshow(img.permute(1, 2, 0))

    gradient = np.linspace(0, 1, len(coord_x))
    plt.scatter(coord_x, coord_y, c=gradient, cmap='viridis', s=40)
    if show_plt:
        plt.show()

    if save_path:
        plt.savefig(save_path)

    if not show_plt:
        plt.close(fig)

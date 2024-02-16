import torch
import torch.nn as nn
from torchvision import transforms


class RandomCutOut(nn.Module):
    def __init__(self, size=(16, 16), min_cutout=0.2, max_cutout=0.8):
        '''
        applies cutout on at least 1x1 pixel. use RandomApply to choose whether this transforms is applied

        @params:
          size: size of input image, (y: num rows,x: num cols)
          max_cutout: max size in terms of % cut out from final patch
        '''
        super().__init__()

        self.size = size

        # (y: num rows,x: num cols)
        self.min_cutout = (int(min_cutout*size[0]), int(min_cutout*size[1]))
        # make sure at least 1 pixel
        self.min_cutout = (
            max(1, self.min_cutout[0]), max(1, self.min_cutout[1]))

        # (y: num rows,x: num cols)
        self.max_cutout = (int(max_cutout*size[0]), int(max_cutout*size[1]))

    def forward(self, img):
        '''
        @params:
          img: assume shape (Nc, Height, Width)
        '''

        y_cutout, x_cutout = torch.randint(self.min_cutout[0], self.max_cutout[0], (1,))[
            0], torch.randint(self.min_cutout[0], self.max_cutout[1], (1,))[0]
        y_start, x_start = torch.randint(
            self.size[0]-y_cutout, (1,))[0], torch.randint(self.size[1]-x_cutout, (1,))[0]

        img[:, y_start:y_start + y_cutout, x_start: x_start+x_cutout] = 0.0
        return img


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

'''
- create functions that are similar to how Pytorch's nn work. We can stack different filtering techniques to filter our timeseries data
- NOTE: assume all input files regardless from whichever machine are csv
- NOTE: all eye-tracking/gaze csv have same format, ie name for columns, ie x_norm
- if gaze data is periodic can just fourier transform -> low pass filter

@input:
    df: df for a given eye-tracking file on a single file. 
        - duration | x_norm | y_norm

'''

from typing import Any
import warnings
import numpy as np


class Filter():
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        warnings.WarningMessage('Implement forward pass')


########################################################################################################
#                                       Dropna
########################################################################################################
class Nan_Filter(Filter):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, df) -> Any:
        '''
        @params
            data: csv file
        '''
        return df.dropna()


########################################################################################################
#                                       Quantile Filter
########################################################################################################
def quantile_filter(df, quantile, column):
    '''
    @params:
        df: csv file
        column: str, str name
        quantile: 0<= quantile <= 1, everything below quantile gets throw away
    '''

    # return columns in df where values in value are greate than quantile of values in column
    # ie column is gaze duration, return columns in whole df where gaze duration >= 25% quantile
    return df[df[column] >= df[column].quantile(quantile)]


class Quantile_Filter(Filter):
    def __init__(self, quantile, column) -> None:
        super().__init__()
        self.quantile = quantile
        self.column = column

    def __call__(self, df, quantile) -> Any:
        '''
        @params
            data: csv file
        '''
        return quantile_filter(df, self.quantile, self.column)


########################################################################################################
#                                      Up/Down Sampling
########################################################################################################

class Sampling_Filter(Filter):
    '''
    This filter does two things:
        - Upsample: bin a fixed duration of fixations together every x interval, then average the positions amongst the bins
            - if a fixation is in between two different bins, put the leftout part into next bin
        - Downsample: split positions with duration > x interval into separate bins
    '''

    def __init__(self, interval) -> None:
        '''
        @param:
            - interval: [ms]
        '''
        super().__init__()
        self.interval = interval

    @staticmethod
    def _compute_avg_position(bin_curr):
        '''bin_curr [(dur, x_norm , y_norm)]'''
        x_norm_avg = sum([x for (_, x, _) in bin_curr]) / len(bin_curr)
        y_norm_avg = sum([y for (_, _, y) in bin_curr]) / len(bin_curr)
        return x_norm_avg, y_norm_avg

    @staticmethod
    def _reset_bin_curr():
        return [], 0.0

    @staticmethod
    def _all(df_sampled, interval, fix, bin_curr, bin_dur):
        '''
        helper to deal with 

        - case 1: duration is longer than interval
        - case 2: we put current row into a running bin

        we do this by splitting the fixation block into different blocks

        @params
            fix: list, [duration, x_norm, y_norm]

        '''
        # split into chunks that fill up whole interval and overflow ones
        # ie for every fix it can be split into three parts: fill in curr block| blocks of length interval | overflow but not enough to fill a new block
        # fix not need have all of them all the time

        block_fill = fix[0]
        # if overflow, then use take a portion out
        if fix[0] >= interval:
            block_fill = interval - bin_dur

        # number of full blocks in btw
        num_block_full = max(int((fix[0]-block_fill)//interval), 0)
        block_overflow = (fix[0]-block_fill)-num_block_full*interval

        # ========================== now add the fixations ==========================
        #   start with block_fill
        bin_dur += block_fill
        bin_curr.append((block_fill, fix[1], fix[2]))
        if bin_dur >= interval:  # this has to be equal, but >= just in case for floating point errors
            x_norm_avg, y_norm_avg = Sampling_Filter._compute_avg_position(
                bin_curr)
            df_sampled.append([x_norm_avg, y_norm_avg])
            bin_curr, bin_dur = Sampling_Filter._reset_bin_curr()

        #   full blocks
        for i in range(num_block_full):
            df_sampled.append([fix[1], fix[2]])

        #   overflow
        if block_overflow > 0.0:
            bin_dur = block_overflow
            bin_curr.append((block_overflow, fix[1], fix[2]))

        return df_sampled, bin_curr, bin_dur

    def sampling_filter(self, df, df_sampled):
        # use a running list of (dur, x, y) to keey track of curr bin;
        # also bin_dur current length of bin for quick comparsion
        bin_curr, bin_dur = [], 0.0

        # overflowed row from previous, [dur,x,y], 0.0 denotes no prev
        fix_prev = [0.0, 0.0, 0.0]

        for idx, row in df.iterrows():
            # make sure row has all these columns
            fix = None
            try:
                fix = [row.duration, row.x_norm, row.y_norm]
            except:
                raise Exception(
                    "input df must have duration, x_norm and y_norm columns")
            # keep filling in the bins while we still have overflowed
            df_sampled,  bin_curr, bin_dur = Sampling_Filter._all(
                df_sampled, self.interval, fix, bin_curr, bin_dur)

        return df_sampled

    def __call__(self, df) -> Any:
        '''
        @params
            data: csv file
        '''
        df_sampled = []
        return self.sampling_filter(df, df_sampled)


########################################################################################################
#                                      Uniform_Quantize_Filter
########################################################################################################
class Uniform_Quantize_Filter(Filter):
    '''
    filter to split the underlying image/eye-tracking region into 2D grids size ie size = (32,16)
    then from left to right, 0 to biggest, basically find 2D index, then convert to 1D ;)

    @param:
        size: (num col/x, num row/y) or dict {'RLS': (16,32)} for example
    '''

    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    @staticmethod
    def _helper_size(size, rpt_name):
        if rpt_name.split('_')[0] in size:
            return size[rpt_name.split('_')[0]]
        else:
            return size['view']

    @staticmethod
    def _quantize(coord, size):
        '''return x, y'''
        coord_new = int(coord[0]*size[0]), int(coord[1]*size[1])
        return coord_new[1]*size[0]+coord_new[0]

    @staticmethod
    def quantize(df, size, rpt_name=None):
        '''
        @params:
            size: (row, col)
        '''
        if rpt_name and isinstance(size, dict):
            size = Uniform_Quantize_Filter._helper_size(size, rpt_name)

        word_idx = []
        for idx, row in df.iterrows():
            x_ = int(row.x_norm*size[0])  # x is col/horizontal dir
            # technical caviat, if row.y_norm == 1 & let 6 cols 5 rows , then 1*5=5, then to convert 5*6+_ we already have 30,
            # which is more than total number of cells. this is bc we are using 0 idxing and with int(), bound for last row is
            # 4<= y < 5. if we are at 5 it counts as new row
            y_ = int(min(row.y_norm, 0.99999999) * size[1])

            flat_idx = y_*size[0]+x_ - 1
            word_idx.append(flat_idx)  # -1 to convert back to 0 index
            # make sure conversion is correct
            if flat_idx >= size[0]*size[1]:
                print(
                    f' There is a mistake: {row.x_norm} {row.y_norm} {x_} {y_} {y_*size[0]+x_}')
        df['word_idx'] = word_idx
        return df

    def __call__(self, df, curr_rpt_name=None) -> Any:
        '''
        @params
            data: csv file
        '''
        return Uniform_Quantize_Filter.quantize(df, self.size, curr_rpt_name)


########################################################################################################
#                                      Region Split
########################################################################################################
class Region_Filter(Filter):
    '''

    @param:
        size: (num rows, num cols) or dict {'RLS': (16,32)} for example
    '''

    def __init__(self, box, grid_size, box_grid_size={}, img2flip={}) -> None:
        super().__init__()
        self.box = box
        self.grid_size = grid_size
        self.box_grid_size = box_grid_size
        self.img2flip = img2flip

        # compute vocab size
        self.vocab_size = 0  # exclude the 3 st, end and empty
        for r in box_grid_size:
            self.vocab_size += box_grid_size[r][0]*box_grid_size[r][1]

    @staticmethod
    def classify_region(coord, box_size):
        # look at which coord (x,y) is in; else none
        for region in box_size:
            st = (box_size[region][0][0], box_size[region][0][1])
            ed = (box_size[region][0][0] + box_size[region][1][0],
                  box_size[region][0][1] + box_size[region][1][1])
            if st[0] <= coord[0] < ed[0] and st[1] <= coord[1] < ed[1]:
                return region
        return None

    @staticmethod
    def quantize_within_region(coord, grid_size, box_size_r, flip=False):
        '''
        @params;
            coord: (x,y) norm
            grid_size: (12,6) for example, grid for current region
            box_size_r: box_size same as below but for current region
        '''
        # first center current coord
        coord_centered = (coord[0]-box_size_r[0][0],
                          coord[1] - box_size_r[0][1])
        # flip HORIZONTALLY only if needed
        if flip:
            coord_centered = (box_size_r[1][0] -
                              coord_centered[0], coord_centered[1])
        # now coord is within the (0.6,0.6) defined in box_size_r[1], norm size of grid; figure out norm coord in this grid
        coord_centered_norm = (
            coord_centered[0]/box_size_r[1][0], coord_centered[1]/box_size_r[1][1])
        # assign to nearest int, but keep within range; make sure if grid_size (12,6), for x we do go up to 12, since [0,..,11]

        coord_within_grid = min(round(coord_centered_norm[0]*grid_size[0]), grid_size[0]-1), min(
            round(coord_centered_norm[1]*grid_size[1]), grid_size[1]-1)

        return coord_within_grid[1]*grid_size[0] + coord_within_grid[0]

    @staticmethod
    def split_region(df, box_size, grid_size, box_grid_size={}, img2flip={}, vocab_size=np.nan):
        '''
        label each row with region character
        TODO: have another option that choose 

        we can either 1) given a grid_size, split into respective regions 2) we give the grid size ourselves

        @params
            box_size: dict['region name ie A'] = [ start:(x,y), size:(x,y) ]
            grid_size: (portion of img x, poriton of img y)
            mode: how to split each region, 
            flip: which one of the scans to flip right to left, so OD and OS are in the same orientation {'rnfl', 'rgcp',...}
        '''

        if len(box_grid_size) == 0:
            # for each region, we have start and size in both directions; we also have size of grid
            # but so we can first figure out how many grids given target grid size size ie (16,16)
            box_grid_size = {}
            for region in box_size:
                b = box_size[region]
                box_grid_size[region] = (
                    round((b[1][0])/grid_size[0]),  # round to nearest int
                    round((b[1][1])/grid_size[1])
                )

        '''
        else:
            # if box_grid_size given, we want to compute how big the box is
            # NOTE: we are doing this so we dont have to rewrite quantize_within_region
        '''

        # compute offset for each region, ie A start 0, B start at 16*16 if box_grid_size[A] = (16,16) & 0 index
        grid_start_idx = {}
        start = 0
        for region in box_grid_size:
            grid_start_idx[region] = start
            start += box_grid_size[region][0]*box_grid_size[region][1]

        region, region_idx = [], []
        for idx, row in df.iterrows():
            r = Region_Filter.classify_region(
                (row.x_norm, row.y_norm), box_size)

            # current fixation not in a region; append -1
            if not r:
                region.append(np.nan)
                region_idx.append(vocab_size)
            else:
                region.append(r)
                region_idx.append(
                    grid_start_idx[r] + Region_Filter.quantize_within_region(
                        (row.x_norm, row.y_norm), box_grid_size[r], box_size[r], flip=r in img2flip)
                )
        df['region'] = region
        df['region_word_idx'] = region_idx

        return df

    def __call__(self, df) -> Any:
        '''
        @params
            data: csv file
        '''
        return Region_Filter.split_region(df, self.box, self.grid_size, self.box_grid_size, self.img2flip, vocab_size=self.vocab_size+1)


########################################################################################################
#                                      Drop negative values by column
########################################################################################################
class NegativeValuesFilter(Filter):
    def __init__(self, x_column_name='x_norm', y_column_name='y_norm') -> None:
        super().__init__()
        self.x_column_name = x_column_name
        self.y_column_name = y_column_name

    def filter_negative_values(self, data):
        # Assuming 'data' is a pandas DataFrame or a similar data structure
        # Remove rows with negative values in x_norm and/or y_norm columns
        filtered_data = data[(data[self.x_column_name] >= 0)
                             & (data[self.y_column_name] >= 0)]
        return filtered_data

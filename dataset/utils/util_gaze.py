'''base on pygazeanalyser, modify their code for our use. NOTE: right now it onlys works with pavlovia'''
import sys
import numpy as np
import os
from matplotlib import pyplot, image
import matplotlib
import torch

# append path to pygaze
sys.path.append('/../../util_packages/PyGazeAnalyser/')
# from pygazeanalyser.gazeplotter import  draw_display


def parse_fixations(df, dispsize):
    fix = {
        'x': np.zeros(len(df)),
        'y': np.zeros(len(df)),
        'dur': np.zeros(len(df))
    }
    for idx, row in df.iterrows():
        fix['x'][idx] = row.norm_pos_x*dispsize[0]
        fix['y'][idx] = row.norm_pos_y*dispsize[1]
        fix['dur'][idx] = row.duration
    return fix


def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                                    e.g. (width,height)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                                    is to be laid, or None for no image; NOTE: the image
                                    may be smaller than the display size, the function
                                    assumes that the image was presented at the centre of
                                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                                    with a size of dispsize, and an image drawn onto it
                                    if an imagefile was passed
    """

    # construct screen (black background)
    _, ext = os.path.splitext(imagefile)
    ext = ext.lower()
    data_type = 'float32' if ext == '.png' else 'uint8'
    screen = np.zeros((dispsize[1], dispsize[0], dispsize[2]), dtype=data_type)
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception(
                "ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)
        # flip image over the horizontal axis
        # (do not do so on Windows, as the image appears to be loaded with
        # the correct side up there; what's up with that? :/)
        if not os.name == 'nt':
            img = np.flipud(img)
        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0]/2 - w/2)
        y = int(dispsize[1]/2 - h/2)
        # draw the image on the screen
        try:
            screen[y:y+h, x:x+w, :] += img
        except:
            print(f'This image has {img.shape} {imagefile}')
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])

    ax.imshow(screen)  # , origin='upper')

    return fig, ax


def helper_draw_heatmap(
        fixations,
        dispsize,
        imagefile=None,
        durationweight=True,
        alpha=0.5,
        cmap='jet',
        savefilename=None):
    '''
    Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.
    @params:
    fixations		- full df df, assume x, y positions are unnormalized. use dipsize to rescale
    dispsize		- tuple or list indicating the size of the display,
                    e.g. (1024,768)

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    durationweight	-	Boolean indicating whether the fixation duration is
                    to be taken into account as a weight for the heatmap
                    intensity; longer duration = hotter (default = True)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    '''

    # FIXATIONS
    fix = parse_fixations(fixations, dispsize)

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = 200  # guassian width and sd width
    gsdwh = gwh/6
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh/2)
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(fix['dur'])):
        # get x and y coordinates
        # x and y - indexes of heatmap array. must be integers
        x = strt + int(fix['x'][i]) - int(gwh/2)
        y = strt + int(fix['y'][i]) - int(gwh/2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x-dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y-dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y+vadj[1], x:x+hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * fix['dur'][i]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y+gwh, x:x+gwh] += gaus * fix['dur'][i]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1]+strt, strt:dispsize[0]+strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = np.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)

    # FINISH PLOT
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)
    pyplot.close()
    return fig


def gaussian(x, sx, y=None, sy=None):
    """Returns an array of np arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    @params
    x		-- width in pixels
    sx		-- width standard deviation	
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo, yo = x/2, y/2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)
                                      ) + ((float(j)-yo)**2/(2*sy*sy))))
    return M

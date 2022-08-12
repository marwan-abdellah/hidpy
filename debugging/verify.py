import os 
import imageio
from pathlib import Path
from matplotlib.pyplot import show
from numpy import angle, arctan2
from pyoptflow import HornSchunck
from pyoptflow.plots import compareGraphs
import argparse
from matplotlib.pyplot import figure, draw, pause, gca
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize



####################################################################################################
# @get_files_in_directory
####################################################################################################
def get_files_in_directory(directory,
                           file_extension=None) -> list:

    # A list of all the files that exist in a directory
    files = list()

    # If the extension is not specified
    if file_extension is None:
        for i_file in os.listdir(directory):
            files.append(i_file)

    # Otherwise, return files that have specific extensions
    else:
        for i_file in os.listdir(directory):
            if i_file.endswith(file_extension):
                files.append(int(i_file.strip('.%s' % file_extension)))

    # Sort to ensure that you get the consequentive frames 
    files.sort()

    # Return the list
    return files


####################################################################################################
# @parse_command_line_arguments
####################################################################################################
def parse_command_line_arguments():

    # add all the options
    description = 'Pure Python Horn Schunck Optical Flow'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'Input sequence'
    parser.add_argument('--input-sequence-path', '-i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artefacts will be stored'
    parser.add_argument('--output-directory', '-o', action='store', help=arg_help)

    arg_help = 'Show all the resulting ploys on-the-fly'
    parser.add_argument('--plot', help=arg_help, action='store_true')

    arg_help = 'The regulization parameter, the default value is 0.001'
    parser.add_argument('--alpha', help=arg_help, type=float, default=0.001)

    arg_help = 'Number of iterations, default 8'
    parser.add_argument("--iterations", help=arg_help, type=int, default=8)

    # Parse the arguments
    return parser.parse_args()



def plot_quiver(ax, flow, spacing, margin=0, name='image',**kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    h, w, *_ = flow.shape

    nx = int((w - 2 * margin) / spacing) 
    ny = int((h - 2 * margin) / spacing)

    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    ax.quiver(x, y, u, v, **kwargs)

    ax.set_ylim(sorted(ax.get_ylim(), reverse=False))
    ax.set_aspect("equal")
    plt.savefig('%s.png' % name)
    plt.close()



def get_pixel_center(image, i_pixel, j_pixel):

    pass  


def get_pixel_indices(i_pixel, j_pixel, dx, dy):

    pass 








####################################################################################################
# @compute_optical_flow_hs
####################################################################################################
def compute_optical_flow_hs(args):

    extension = 'bmp'

    # Get all the images in the directory sorted 
    file_list = get_files_in_directory(args.input_sequence_path, extension)


   

####################################################################################################
# @Main
####################################################################################################
if __name__ == "__main__":

    import numpy as np
    import cv2 as cv
    import argparse
    
    cap = cv.VideoCapture('/projects/py-hi-d/data/sequence1/video/sequence1.avi')
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.001,
                        minDistance = 0.002,
                        blockSize = 2 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    ii = 0
    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
        
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 1)
            frame = cv.circle(frame, (int(a), int(b)), 1, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)


        number  = str(ii)

        ff = '%s.png' % number.zfill(3)
        cv.imwrite(ff, img)
        ii += 1

        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()
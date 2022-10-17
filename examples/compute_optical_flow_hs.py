from __future__ import annotations
from scipy.signal import convolve2d
import numpy as np

import os
from random import random
from time import sleep 
import imageio
from pathlib import Path
from matplotlib.pyplot import show
from numpy import angle, arctan2
#from pyoptflow.plots import compareGraphs
import argparse
from matplotlib.pyplot import figure, draw, pause, gca
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cv2
import copy 







HSKERN = np.array(
    [[1 / 12, 1 / 6, 1 / 12], 
    [1 / 6, 0, 1 / 6], 
    [1 / 12, 1 / 6, 1 / 12]], float
)

kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx

kernelY = np.array([[-1, -1], [1, 1]]) * 0.25  # kernel for computing d/dy

kernelT = np.ones((2, 2)) * 0.25


def computeDerivatives(
    im1: np.ndarray, im2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    fx = convolve2d(im1, kernelX, "same") + convolve2d(im2, kernelX, "same")
    fy = convolve2d(im1, kernelY, "same") + convolve2d(im2, kernelY, "same")

    # ft = im2 - im1
    ft = convolve2d(im1, kernelT, "same") + convolve2d(im2, -kernelT, "same")

    return fx, fy, ft

def HornSchunck(
    im1: np.ndarray,
    im2: np.ndarray,
    *,
    alpha: float = 0.001,
    Niter: int = 8,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------

    im1: numpy.ndarray
        image at t=0
    im2: numpy.ndarray
        image at t=1
    alpha: float
        regularization constant
    Niter: int
        number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)
    vInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    '''
    if verbose:
        from .plots import plotderiv
        plotderiv(fx, fy, ft)
    '''

    # Iteration to reduce error
    for _ in range(Niter):
        # %% Compute local averages of the flow vectors
        uAvg = convolve2d(U, HSKERN, "same")
        vAvg = convolve2d(V, HSKERN, "same")
        # %% common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        # %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V




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


import scipy 
import time 


def compute_interpolated_flow_fields(Us, Vs):

    fUs = list()
    fVs = list()

    xx = np.arange(Us[0].shape[0])
    yy = np.arange(Vs[0].shape[1])

    for t in range(len(Us)):

        fu = scipy.interpolate.interp2d(xx, yy, Us[t], kind='cubic')
        fv = scipy.interpolate.interp2d(xx, yy, Vs[t], kind='cubic')

        fUs.append(fu)
        fVs.append(fv)
    
    return fUs, fVs


def compute_trajectory(x0, y0, fUs, fVs):

    x_current = x0 
    y_current = y0

    trajectory = list()

    # Get the current pixel 
    for t in range(len(fUs)):
        
        #t0 = time.time()
        #xx = np.arange(Us[t].shape[0])
        #yy = np.arange(Vs[t].shape[1])
        #fu = scipy.interpolate.interp2d(xx, yy, Us[t], kind='cubic')
        #fv = scipy.interpolate.interp2d(xx, yy, Vs[t], kind='cubic')

        dx = fUs[t](x_current, y_current)
        dy = fVs[t](x_current, y_current)
        #t1 = time.time()
        
        # dx = Us[t][x_current, y_current]
        # dy = Vs[t][x_current, y_current]

        x_new = x_current + dx
        y_new = y_current + dy

        # Add the x_pixel and y_pixel to the list 
        trajectory.append([x_new, y_new])

        x_current = (x_new)
        y_current = (y_new)

    return trajectory


####################################################################################################
# @create_numpy_padded_image
####################################################################################################
def create_numpy_padded_image(image_path):
    
    # Create image object 
    loaded_image = imageio.imread(image_path, as_gray=True)
    
    # Image size 
    image_size = loaded_image.shape
    
    # Make a square image 
    square_size = image_size[0]
    if image_size[1] > image_size[0]:
        square_size = image_size[1]
    
    # Ensure that it is even 
    square_size = square_size if square_size % 2 == 0 else square_size + 1
    
    # Create a square image 
    square_image = Image.new(mode='L', size=(square_size, square_size), color='black')    
    square_image.paste(Image.fromarray(np.float32(loaded_image)))
    square_image = np.float32(square_image)
        
    # Return the square image 
    return square_image


####################################################################################################
# @compute_optical_flow_hs
####################################################################################################
def compute_optical_flow_hs(args):

    extension = 'bmp'

    # Get all the images in the directory sorted 
    file_list = get_files_in_directory(args.input_sequence_path, extension)

    # Displacement arrays 
    u_arrays = list()
    v_arrays = list()

    # Compute the optical flow frames
    for i in range(len(file_list) - 1):
        
        # The first frame         
        frame1_path = '%s/%s.%s' % (args.input_sequence_path, file_list[i], extension)
        frame_1 = create_numpy_padded_image(frame1_path)

        # The second frame         
        frame2_path = '%s/%s.%s' % (args.input_sequence_path, file_list[i + 1], extension)
        frame_2 = create_numpy_padded_image(frame2_path)
        
        # Run the optical flow method
        V, U = HornSchunck(frame_1, frame_2, alpha=0.001, Niter=1)

        # Append the flow fields to the arrays 
        u_arrays.append(U)
        v_arrays.append(V)

    # The first frame         
    frame1_path = '%s/%s.%s' % (args.input_sequence_path, file_list[i], extension)
    frame_1 = create_numpy_padded_image(frame1_path)

    fUs, fVs = compute_interpolated_flow_fields(u_arrays, v_arrays)

    trajectories = list()
    for ii in range(frame_1.shape[0]):
        for jj in range(frame_1.shape[1]):
            if frame_1[ii, jj] > 15:
                print(ii, jj)
                trajectories.append(compute_trajectory(ii, jj, fUs, fVs))


    import random
    #trajectories =random.sample(trajectories, 1)
    
    xrgb = Image.fromarray(frame_1).convert("RGB")
    xnp = np.array(xrgb)

    for i, traj in enumerate(trajectories):
        print(i, len(trajectories))

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        cv2.circle(xnp, (int(traj[0][1]), int(traj[0][0])), 1, (r,g,b), 1)

        for kk in range(len(traj) - 1):
            
            y0 = int(traj[kk][0])
            x0 = int(traj[kk][1])

            y1 = int(traj[kk + 1][0])
            x1 = int(traj[kk + 1][1])

            
            cv2.line(xnp, (x0,y0), (x1,y1), (r,g,b), 1)
        
    cv2.imwrite('trajectory.png', xnp)

####################################################################################################
# @Main
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()
    

    '''
    from scipy import interpolate
    x = np.arange(-10, 5.01, 0.01)
    y = np.arange(-10, 5.01, 0.01)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+yy**2)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    import matplotlib.pyplot as plt
    xnew = np.arange(-10, 10, 1e-2)
    ynew = np.arange(-10, 10, 1e-2)
    znew = f(xnew, ynew)
    plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
    plt.show()
    '''

    # Compute the optical flow using the HS method
    U, V = compute_optical_flow_hs(args)

    #show()


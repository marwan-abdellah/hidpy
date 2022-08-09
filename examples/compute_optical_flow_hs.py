import os
from random import random
from time import sleep 
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
import cv2
import copy 


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



def compute_trajectory(x0, y0, fUs, fVs, pixel_size=1):

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

def verify_flow_frame(frame_0, frame_1, U, V, pixel_width=0.002688172043010753):

    width = frame_0.shape[0]
    height = frame_0.shape[1]

    for ii in range(width):
        for jj in range(height):

            x0 = ii
            y0 = jj

            v0 = frame_0[x0, y0]

            import math
            x1 = int(x0 + U[ii, jj] + 0.5 * pixel_width)
            y1 = int(y0 + V[ii, jj] + 0.5 * pixel_width)

            
            v1 = frame_1[x0, y0]

            if v0 > 120:
                print(x0, y0, '', x1, y1, '', v0, v1)

                #print(v0, v1)
                sleep(0.001)



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

        #flow = np.dstack((U, V)) 
        #fig, ax = plt.subplots()

        #pixel_size = 1.0 / frame0.shape[1]
        #plot_quiver(ax, flow,  spacing=5, margin=1, name=str(i), color="#ff34ff")

        #verify_flow_frame(im1, im2, U, V)

    # The first frame         
    frame1_path = '%s/%s.%s' % (args.input_sequence_path, file_list[i], extension)
    frame_1 = create_numpy_padded_image(frame1_path)

    fUs, fVs = compute_interpolated_flow_fields(u_arrays, v_arrays)

    trajectories = list()
    for ii in range(frame_1.shape[0]):
        for jj in range(frame_1.shape[1]):
            if frame_1[ii, jj] > 10:
                print(ii, jj)
                trajectories.append(compute_trajectory(ii, jj, fUs, fVs))


    import random
    #trajectories =random.sample(trajectories, 2)
    
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

    
    

    exit(0)








    '''
    mask = np.random.randint(255, size=(U.shape[0] * U.shape[1]),dtype=np.uint8)
    mask = mask.reshape(U.shape[0], U.shape[1])
    # mask = Image.new(mode="L", size=(U.shape[0], U.shape[1]))


    
    
    ss = list()
    for ii in range(U.shape[0]):
        for jj in range(U.shape[1]):
            valueU = U[ii, jj]
            valueV = V[ii, jj]

            if valueU == 0 or valueV == 0:
                print('zero %d %d' % (ii, jj))
    maskim = Image.fromarray(mask)

    ss = 'fra%d.png' % i
    maskim.save(ss)



    
    print(flow.shape)

    fig, ax = plt.subplots()
    plot_quiver(ax, flow,  spacing=10, scale=10, name=str(i), color="#ff34ff")

    fig, ax =plt.subplots(1,2)
    import seaborn as sns
    sns.heatmap(U, annot=False, ax=ax[0])
    sns.heatmap(V, annot=False, ax=ax[1])
    plt.show()


    #
    if i > 10:
        exit(0)
    else:
        continue

    continue

    
    # print(im1.shape)
    # print(U.shape, V.shape)
    # exit(0)
    
    ph = np.linspace(0, 2*np.pi, 13)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = arctan2(u, v)

    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.plasma

    X = range(im1.shape[0])
    Y = range(im1.shape[1])

    X, Y = np.meshgrid(X, Y)
    print(X)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.quiver(X,Y,U,V, scale=100)
    plt.show()
    exit(0)
    '''

    '''
    ph = np.linspace(0, 2 * np.pi, 128)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(u, v)

    norm = Normalize()
    norm.autoscale(colors)

    print(colors)
    exit(0)
    
    
    scale=1
    quivstep = 10
    ax = figure().gca()
    #ax.imshow(Inew, cmap="gray", origin="lower")
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    for i in range(0, U.shape[0], quivstep):
        for j in range(0, V.shape[1], quivstep):


            angle = arctan2(V[i, j], U[i, j])
            
            ax.arrow(
                j,
                i,
                V[i, j] * scale,
                U[i, j] * scale,
                #color=colors(),
                head_width=0.5,
                head_length=1,
            )
    plt.show()
    


    exit(0)
    '''
    # Compare the Graphs 
    # compareGraphs(U, V, im2, fn=file_list[i + 1])

    #return U, V


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


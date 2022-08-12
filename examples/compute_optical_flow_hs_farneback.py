from pickletools import pylist
import cv2
import numpy as np
import argparse
import time
from sys import stdout
import matplotlib.pyplot as plt

from time import sleep


####################################################################################################
# @parse_command_line_arguments
####################################################################################################
def parse_command_line_arguments():

    # add all the options
    description = 'This application takes an input sequence and computes the intensity profile '
    'along the entire sequence'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'Input sequence'
    parser.add_argument('--input-sequence', '-i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artefacts will be stored'
    parser.add_argument('--output-directory', '-o', action='store', help=arg_help)

    # Parse the arguments
    return parser.parse_args()


####################################################################################################
# @get_frame
####################################################################################################
def get_frame(video_capture, frame_number):

    # Set the video to the specific frame 
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Check if the video has frames or not and get a list 
    has_frame, frame = video_capture.read()

    if has_frame:
        return frame
    else:
        print('Frame [%d] does not exist, NOT grabbed!' % frame_number)
        exit(0)

def plot_quiver(ax, flow, spacing, margin=0, **kwargs):
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
    plt.savefig('hola.png')


def compute_trajectory(x0, y0, Us, Vs, pixel_size=1):

    x_current = x0 
    y_current = y0

    trajectory = list()

    # Get the current pixel 
    for t in range(len(Us)):
        
        dx = Us[t][x_current, y_current]
        dy = Vs[t][x_current, y_current]

        x_new = x_current + dx 
        y_new = y_current + dy 


        #print(x_current, y_current)
        


        x_pixel_new = int(x_new)
        y_pixel_new = int(y_new)

        #print(dx, dy)
        #print('----')
        #sleep(1)


        # Add the x_pixel and y_pixel to the list 
        trajectory.append([x_pixel_new, y_pixel_new])

        x_current = x_pixel_new
        y_current = y_pixel_new

    #print(trajectory)
    
    return trajectory

from PIL import Image 



def verify_flow_frame(frame_0, frame_1, U, V, pixel_width=1):#0.002688172043010753):

    width = frame_0.shape[0]
    height = frame_0.shape[1]

    for ii in range(width):
        for jj in range(height):

            x0 = ii
            y0 = jj

            v0 = frame_0[x0, y0]

            x1 = int(x0 + U[ii, jj] / pixel_width)
            y1 = int(y0 + V[ii, jj] / pixel_width)

            
            v1 = frame_1[x0, y0]

            if v0 > 120:
                print(x0, y0, '', x1, y1, '', v0, v1)

                #print(v0, v1)
                sleep(0.001)

####################################################################################################
# @compute_optical_flow_franeback
####################################################################################################
def compute_optical_flow_franeback(args):

    # Read the video anc create the VideoCapture object 
    video_capture = cv2.VideoCapture(args.input_sequence)

    # Get the number of frames in the video 
    number_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the first frame and compute its gray scale 
    frame_0 = get_frame(video_capture=video_capture, frame_number=0)

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame_0)

    
    # Sets image saturation to maximum
    mask[..., 1] = 255

    frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

    im_pil = Image.fromarray(frame_0)


    u_arrays = list()
    v_arrays = list()

    # Process the sequence, frame[i] and frame[i + 1]
    for i in range(1, number_frames - 1):
        
        stdout.write('OF Frames [%d - %d] \n' % (i, i + 1))
        stdout.flush()

        # Get the second frame and compute its gray scale 
        frame_1 = get_frame(video_capture=video_capture, frame_number=i)        
        frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('Input Frame 1', frame_0)
        #k = cv2.waitKey(100) & 0xff

        #cv2.imshow('Input Frame 2', frame_1)
        #k = cv2.waitKey(100) & 0xff
        
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            frame_0, frame_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        U = flow[:,:,0]
        V = flow[:, :, 1]

        u_arrays.append(U)
        v_arrays.append(V)


        #verify_flow_frame(frame_0, frame_1, U, V)


        frame_0 = frame_1

        

    print('computing trajectories')

    trajectories = list()
    for ii in range(frame_0.shape[0]):
        print(ii, frame_0.shape[0])
        for jj in range(frame_0.shape[1]):
            if frame_0[ii, jj] > 10:
                trajectories.append(compute_trajectory(ii, jj, u_arrays, v_arrays))

    import random
    
    print('sampling')
    trajectories =random.sample(trajectories, 25)



    xrgb = im_pil.convert("RGB")
    xnp = np.array(xrgb)

    

    for i, traj in enumerate(trajectories):
        print(i, len(trajectories))
        
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        cv2.circle(xnp, (traj[0][1], traj[0][0]), 2, (r,g,b), 2)

        for kk in range(len(traj) - 1):
            
            y0 = traj[kk][0]
            x0 = traj[kk][1]

            y1 = traj[kk + 1][0]
            x1 = traj[kk + 1][1]

            
            cv2.line(xnp, (x0,y0), (x1,y1), (r,g,b), 1)
        
    cv2.imwrite('trajectory.png', xnp)
    '''
        #fig, ax = plt.subplots()
        #plot_quiver(ax, flow, spacing=5, scale=0.05, color="#ff34ff")
        #time.sleep(1)
        #exit(0)



        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        cv2.imshow('Dense Optical Flow', rgb)
        k = cv2.waitKey(100) & 0xff

        frame_0 = frame_1.copy()

    # Free up resources and closes all windows
    video_capture.release()
    cv2.destroyAllWindows()
    '''

####################################################################################################
# @Main
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()
    
    # Compute the intensity profile 
    compute_optical_flow_franeback(args)
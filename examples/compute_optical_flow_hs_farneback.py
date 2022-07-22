import cv2
import numpy as np
import argparse
import time
from sys import stdout


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

    # Process the sequence, frame[i] and frame[i + 1]
    for i in range(1, number_frames - 1):
        
        stdout.write('OF Frames [%d - %d] \n' % (i, i + 1))
        stdout.flush()

        # Get the second frame and compute its gray scale 
        frame_1 = get_frame(video_capture=video_capture, frame_number=i)        
        frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Input Frame 1', frame_0)
        k = cv2.waitKey(100) & 0xff

        cv2.imshow('Input Frame 2', frame_1)
        k = cv2.waitKey(100) & 0xff
        
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            frame_0, frame_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        # Opens a new window and displays the output frame
        # cv2.imshow("Dense Optical Flow", rgb)
        
        cv2.imshow('Dense Optical Flow', rgb)
        k = cv2.waitKey(100) & 0xff

        frame_0 = frame_1.copy()

    # Free up resources and closes all windows
    video_capture.release()
    cv2.destroyAllWindows()
    

####################################################################################################
# @Main
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

    # Compute the intensity profile 
    compute_optical_flow_franeback(args)
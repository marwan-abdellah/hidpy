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
# @compute_optical_flow_lk
####################################################################################################
def compute_optical_flow_lk(args):



    # Adjust the parameters for corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    
    # Adjust the parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Read the video anc create the VideoCapture object 
    video_capture = cv2.VideoCapture(args.input_sequence)

    # Get the number of frames in the video 
    number_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the first frame and compute its gray scale 
    frame_0 = get_frame(video_capture=video_capture, frame_number=0)

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame_0)

    # Compute the grayscale image of the first frame 
    frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

    # Finding corners based on good features to track 
    p0 = cv2.goodFeaturesToTrack(frame_0, mask=None, **feature_params)
                                
    # Process the sequence, frame[i] and frame[i + 1]
    for i in range(1, number_frames - 1):
        
        stdout.write('OF Frames [%d - %d] \n' % (i, i + 1))
        stdout.flush()

        # Get the second frame and compute its gray scale 
        frame_1 = get_frame(video_capture=video_capture, frame_number=i)        
        frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Input Frame 1', frame_0)
        k = cv2.waitKey(100) & 0xff

        cv2.imshow('Input Frame 2', frame_1_gray)
        k = cv2.waitKey(1000) & 0xff

        # Calculate optical flow by the Lucas Kandas method 
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame_0, frame_1_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            
            a, b = new.ravel()
            c, d = old.ravel()

            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

            frame_1 = cv2.circle(frame_1, (a, b), 5, color[i].tolist(), -1)
          
        flow_image = cv2.add(frame_1, mask)
        k = cv2.waitKey(100) & 0xff

        # Updating the next frame 
        frame_0 = frame_1_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

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
    compute_optical_flow_lk(args)
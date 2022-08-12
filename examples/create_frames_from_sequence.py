# Imports 
import argparse
import cv2
from sys import stdout


####################################################################################################
# @parse_command_line_arguments
####################################################################################################
def parse_command_line_arguments():

    # add all the options
    description = 'This application takes an input sequence and creates a set of frames.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'Input sequence'
    parser.add_argument('--input-sequence', '-i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artefacts will be stored'
    parser.add_argument('--output-directory', '-o', action='store', help=arg_help)

    arg_help = 'Image extension'
    parser.add_argument('--image-extension', '-e', action='store', default='bmp',  help=arg_help)

    # Parse the arguments
    return parser.parse_args()


####################################################################################################
# @write_frame_to_file
####################################################################################################
def write_frame_to_file(video_capture, frame_number, output_directory, extension='bmp'):
    
    # Set the video to the specific frame 
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Check if the video has frames or not and get a list 
    has_frame, frame = video_capture.read()
    
    # If the video has the frame write it  
    if has_frame:

        # Adjust the path 
        image_path = '%s/%s.%s' % (output_directory, frame_number, extension)

        # Write the frame into an image 
        cv2.imwrite(image_path, frame)     
    
    # No frame with the specified frame number 
    else:
        print('Frame [%d] does NOT exist in the video' % frame_number)
        exit(0)
    

####################################################################################################
# @convert_video_to_frames
####################################################################################################
def convert_video_to_frames(args):

    # Get the video capture from the video file 
    video_capture = cv2.VideoCapture(args.input_sequence)

    # Get the frame rate 
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Get the number of frames in the video 
    number_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Print the video details 
    string = "Video Details: \n"
    string += "  * Name: %s \n" % args.input_sequence
    string += "  * Number Frames %d \n" % number_frames 
    string += "  * FPS: %f \n" % float(fps)
    print(string)

    # Write the frames to disk 
    for i in range(number_frames):
        stdout.write('\rWriting Frames [%d/%d]' % (i + 1, number_frames))
        stdout.flush()
        write_frame_to_file(video_capture=video_capture,
                            frame_number=i, 
                            output_directory=args.output_directory,
                            extension=args.image_extension)
    stdout.write('\nWriting Frames Done \n\n')

################################################################################
# @ Main
################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

    # Convert the video into a sequence 
    convert_video_to_frames(args)
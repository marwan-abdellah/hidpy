# Imports 
import argparse
from sys import prefix
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


####################################################################################################
# Per-adjust all the plotting configuration
####################################################################################################
font_size = 30
plt.rcParams['axes.grid'] = 'True'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.monospace'] = 'Regular'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['figure.titlesize'] = font_size
plt.rcParams['axes.titlesize'] = font_size
plt.rcParams['xtick.major.pad'] = '10'
plt.rcParams['ytick.major.pad'] = '10'
plt.rcParams['axes.edgecolor'] = '1'


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
# @plot_intensities
####################################################################################################
def plot_intensities(intensities, 
                     output_prefix):

    plt.figure(figsize=(15, 10))

    # Min and max values
    min_value = min(intensities)
    max_value = max(intensities)
    major_ticks = np.linspace(min_value, max_value, 4)
    
    # Plot the intensities
    ax = sns.lineplot(data=intensities)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    plt.title('Intensity Profile', y=1.05)
    plt.xlabel('Frames')
    plt.ylabel('Intensity (a.u.)')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))

    # Save the figure 
    plt.savefig('%s.pdf' % output_prefix, dpi=300)


####################################################################################################
# @compute_frame_intensity
####################################################################################################
def compute_frame_intensity(frame):

    # Return the intensity of the given frame based on the numpy array 
    return np.mean(frame)


####################################################################################################
# @decompose_sequence_into_frames
####################################################################################################
def decompose_sequence_into_frames(input_sequence):

    # Get the video capture from the video file 
    video_capture = cv2.VideoCapture(input_sequence)

    # Get the number of frames in the video 
    number_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # A list of all the frames 
    frames = list()
    for i in range(number_frames):
        has_frame, frame = video_capture.read()
        
        if has_frame:
            frames.append(frame)
    
    # Return the list of frames 
    return frames


####################################################################################################
# @compute_frame_intensity
####################################################################################################
def compute_intensity_profile(args):

    # Decompose the sequence into a list of frames 
    frames = decompose_sequence_into_frames(args.input_sequence)

    # Compute the intensity of every frame and save them into a list 
    intensities = list()
    for frame in frames:
        intensities.append(compute_frame_intensity(frame=frame))

    # Plot the intensities 
    plot_intensities(intensities=intensities, 
                     output_prefix='%s/%s' % (args.output_directory, 'intensity'))


####################################################################################################
# @compute_frame_intensity
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

    # Compute the intensity profile 
    compute_intensity_profile(args)
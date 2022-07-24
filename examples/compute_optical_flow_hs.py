import os 
import imageio
from pathlib import Path
from matplotlib.pyplot import show
from pyoptflow import HornSchunck
from pyoptflow.plots import compareGraphs
import argparse

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
                files.append(i_file)

    files = sorted(files) 

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


####################################################################################################
# @compute_optical_flow_hs
####################################################################################################
def compute_optical_flow_hs(args):

    #flist = getimgfiles(stem)
    flist = get_files_in_directory(args.input_sequence_path, 'bmp')
    
    
    for i in range(len(flist) - 1):

        # The first frame         
        fn1 = '%s/%s' % (args.input_sequence_path, flist[i])
        
        # The image of the first frame 
        im1 = imageio.imread(fn1, as_gray=True)

        # The second frame 
        fn2 = '%s/%s' % (args.input_sequence_path, flist[i + 1])
        
        # The image of the second frame 
        im2 = imageio.imread(fn2, as_gray=True)

        # Run the optical flow method [PARAMETERS MUST BE CORRECTED]
        U, V = HornSchunck(im1, im2, alpha=1.0, Niter=100)

        # Compare the Graphs 
        compareGraphs(U, V, im2, fn=flist[i + 1])

    return U, V


####################################################################################################
# @Main
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()
    
    # Compute the optical flow using the HS method
    U, V = compute_optical_flow_hs(args)

    show()


# System imports
import sys
import os
import argparse
from PIL import Image
import numpy as np
import subprocess
import shutil

####################################################################################################
# @parse_command_line_arguments
####################################################################################################
def parse_command_line_arguments(arguments=None):
    """Parses the input arguments.
    :param arguments:
        Command line arguments.
    :return:
        Argument list.
    """

    # add all the options
    description = 'py-hi-d is a pythonic implementation to the technique presented by Shaban et al, 2020.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'The input stack or image sequence that will be processed to generate the output data'
    parser.add_argument('--input-sequence', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artefacts will be stored'
    parser.add_argument('--output-directory', action='store', help=arg_help)

    # Parse the arguments
    return parser.parse_args()

################################################################################
# @ Main
################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    arguments = parse_command_line_arguments()

    dataset = Image.open(arguments.input_sequence)
    
    h,w = np.shape(dataset)
    tiffarray = np.zeros((h,w,dataset.n_frames))
    for i in range(dataset.n_frames):
        dataset.seek(i)
        tiffarray[:,:,i] = np.array(dataset)
        expim = tiffarray.astype(np.double)
    print(expim.shape)
# System imports
import sys
import os
import argparse
import PIL
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
    parser.add_argument('--input-sequenece', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artefacts will be stored'
    parser.add_argument('--output-directory', action='store', help=arg_help)

    # Parse the arguments
    return parser.parse_args()

################################################################################
# @ Main
################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments(sys.argv)


    im = PIL.Image.open(args.input_sequence)
    im.show()

    
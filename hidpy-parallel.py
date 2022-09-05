# System imports
import sys   
import os 
import ntpath
import argparse
import subprocess

# Internal imports 
from core import file_utils

####################################################################################################
# @execute_command
####################################################################################################
def execute_command(shell_command):
    subprocess.call(shell_command, shell=True)


####################################################################################################
# @execute_commands
####################################################################################################
def execute_commands(shell_commands):
    for shell_command in shell_commands:
        print('RUNNING: **********************************************************************')
        print(shell_command)
        print('*******************************************************************************')
        execute_command(shell_command)


####################################################################################################
# @execute_commands_parallel
####################################################################################################
def execute_commands_parallel(shell_commands):
    from joblib import Parallel, delayed
    Parallel(n_jobs=8)(delayed(execute_command)(i) for i in shell_commands)


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
    description = 'hidpy is a pythonic implementation to the technique presented by Shaban et al, 2020.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'The input videos that will be processed to generate the output data'
    parser.add_argument('--input-sequences-directory', '--i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artifacts will be stored'
    parser.add_argument('--output-directory', '--o', action='store', help=arg_help)

    # Parse the arguments
    return parser.parse_args()


####################################################################################################
# @create_shell_commands
####################################################################################################
def create_shell_commands(args):

    # A list of all the commands that will be executed based on the input video sequences
    shell_commands = list()

    # Get a list of all the videos 
    videos = file_utils.get_videos_list(args.input_sequences_directory)

    for video in videos: 

        shell_command = '%s %s/hidpy.py ' % (sys.executable, os.path.dirname(os.path.realpath(__file__)))  
        shell_command += '--input-sequence %s/%s ' % (args.input_sequences_directory, video)
        shell_command += '--output-directory %s/%s' % (args.output_directory, ntpath.basename(video).split('.')[0]) 

        shell_commands.append(shell_command)

    return shell_commands


####################################################################################################
# @__main__
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

    # Compose the commands for parallel execution 
    commands = create_shell_commands(args)

    # Run the jobs in parallel 
    execute_commands_parallel(commands)
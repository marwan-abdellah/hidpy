# System imports
import sys   
import os 
import ntpath
import argparse
import subprocess
import warnings
warnings.filterwarnings('ignore') # Ignore all the warnings 

# Path hidpy
sys.path.append('%s/../' % os.getcwd())

# Internal packages 
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
def execute_commands_parallel(shell_commands, ncores):
    from joblib import Parallel, delayed
    Parallel(n_jobs=ncores)(delayed(execute_command)(i) for i in shell_commands)

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

    description = 'hidpy is a pythonic implementation to the technique presented by Shaban et al, 2020.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'The input configuration file that contains all the data. If this file is provided the other parameters are not considered'
    parser.add_argument('--config-file', action='store', help=arg_help, default='EMPTY')

    arg_help = 'The input stack or image sequence that will be processed to generate the output data'
    parser.add_argument('--input-sequence', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artifacts will be stored'
    parser.add_argument('--output-directory', action='store', help=arg_help)

    arg_help = 'The pixle threshold. (This value should be in microns, and should be known from the microscope camera)'
    parser.add_argument('--pixel-threshold', help=arg_help, type=float, default=10)

    arg_help = 'The pixle size. This value should be tested with trial-and-error'
    parser.add_argument('--pixel-size', help=arg_help, type=float)

    arg_help = 'Number of cores. If 0, it will use all the cores available in the system'
    parser.add_argument('--n-cores', help=arg_help, type=int, default=0)

    arg_help = 'Video time step.'
    parser.add_argument('--dt', help=arg_help, type=float)

    arg_help = 'Use the D model'
    parser.add_argument('--d-model', action='store_true')

    arg_help = 'Use the DA model'
    parser.add_argument('--da-model', action='store_true')

    arg_help = 'Use the V model'
    parser.add_argument('--v-model', action='store_true')

    arg_help = 'Use the DV model'
    parser.add_argument('--dv-model', action='store_true')
    
    arg_help = 'Use the DAV model'
    parser.add_argument('--dav-model', action='store_true')

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
        shell_command += '--output-directory %s' % str(args.output_directory) 
        shell_command += '--pixel-threshold %s' % str(args.pixel_threshold) 
        shell_command += '--pixel-size %s' % str(args.pixel_size) 
        shell_command += '--dt %s' % str(args.dt) 
        shell_command += '--n-cores 1'

        if args.d_model: 
            shell_command += '--d-model'  
        if args.da_model: 
            shell_command += '--da-model'  
        if args.v_model: 
            shell_command += '--v-model'  
        if args.dv_model: 
            shell_command += '--dv-model'  
        if args.dav_model: 
            shell_command += '--dav-model'  
        
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
    execute_commands_parallel(commands, ncores=args.n_cores)
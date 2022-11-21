# System imports  
import argparse

# Internal imports 
from core import optical_flow
from core import video_processing
from core import plotting
from MSD_Bayes_Python import MSDimports
from MSD_Bayes_Python import MSDBayesimports


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

    arg_help = 'The input stack or image sequence that will be processed to generate the output data'
    parser.add_argument('--input-sequence', '-i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artifacts will be stored'
    parser.add_argument('--output-directory', '-o', action='store', help=arg_help)

    arg_help = 'The pixle threshold. (This value should be in microns, and should be known from the microscope camera)'
    parser.add_argument('--pixel-threshold', '-t', help=arg_help, type=float, default=10)

    arg_help = 'The pixle size. This value should be tested with trial-and-error'
    parser.add_argument('--pixel-size', '-s', help=arg_help, type=float)

    arg_help = 'Video time step.'
    parser.add_argument('--delta-t', '-p', help=arg_help, type=float)

    arg_help = 'Number of iterations, default 8'
    parser.add_argument('--iterations', '-n', help=arg_help, type=int, default=8)

    # Models
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
# @__main__
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

    # If configuration exists, then use the configuration 

    # Load the frames from the video 
    print('Loading frames')
    frames = video_processing.get_frames_list_from_video(
        video_path=args.input_sequence, verbose=True)

    # Compute the optical flow
    print('Computing optical flow') 
    u, v = optical_flow.compute_optical_flow_farneback(frames=frames)

    # Interpolate the flow field
    print('Computing interpolations')
    u, v = optical_flow.interpolate_flow_fields(u_arrays=u, v_arrays=v)

    # Compute the trajectories 
    print('Creating trajectories')
    trajectories = optical_flow.compute_trajectories(
        frame=frames[0], fu_arrays=u, fv_arrays=v, pixel_threshold=15)

    # Plot the trajectories 
    print('Plotting trajectories')
    plotting.plot_trajectories_on_frame(
        frame=frames[0], trajectories=trajectories, 
        output_path='%s/trajectories' % args.output_directory)

    print('Saving the trajectories')
    optical_flow.save_trajectories_to_file(trajectories=trajectories,
         file_path='%s/trajectory' % args.output_directory)
    
    # construct trajectory map
    xp, yp = MSDimports.convert_trajectories_to_map(trajectories, (len(frames), frames[0].shape[0], frames[0].shape[1]))

    # extract nucleoli mask
    mask_nuc = MSDimports.extract_nucleoli_map(xp, yp)

    # compute the MSDs
    MSD = MSDimports.MSDcalculation(xp, yp, mask_nuc)

    # Baysian fit on MSDs
    models_selected = ['D','DA','V','DV','DAV'] ### this should be specified by the user in the config file
    dT = 0.1 ### this should be specified by the user in the config file
    Bayes = MSDBayesimports.MSDBayes(MSD, dT, models_selected)

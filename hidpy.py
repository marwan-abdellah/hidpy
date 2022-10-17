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

    # add all the options
    description = 'hidpy is a pythonic implementation to the technique presented by Shaban et al, 2020.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'The input stack or image sequence that will be processed to generate the output data'
    parser.add_argument('--input-sequence', '--i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artifacts will be stored'
    parser.add_argument('--output-directory', '--o', action='store', help=arg_help)

    arg_help = 'Zero-pad the input sequence to make the dimensions along the X and Y the same'
    parser.add_argument('--zero-pad', action='store', help=arg_help)

    arg_help = 'The regulization parameter, the default value is 0.001'
    parser.add_argument('--alpha', '--a', help=arg_help, type=float, default=0.001)

    arg_help = 'Number of iterations, default 8'
    parser.add_argument('--iterations', '--n', help=arg_help, type=int, default=8)

    # Parse the arguments
    return parser.parse_args()


####################################################################################################
# @__main__
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

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

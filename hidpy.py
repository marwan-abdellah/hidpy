# System imports  
import argparse

# Internal imports 
from core import optical_flow
from core import video_processing
from core import plotting


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
    u, v = optical_flow.compute_optical_flow_horn_schunck(frames=frames)

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



















    '''
    dataset = Image.open(arguments.input_sequence)
    
    h,w = np.shape(dataset)
    tiffarray = np.zeros((h,w,dataset.n_frames))
    for i in range(dataset.n_frames):
        dataset.seek(i)
        tiffarray[:,:,i] = np.array(dataset)
        expim = tiffarray.astype(np.double)
    print(expim.shape)
    '''
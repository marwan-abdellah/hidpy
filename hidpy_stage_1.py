# System imports  
import argparse

# System packages 
import os
import numpy
import pathlib 
import sys 
import warnings
import pickle
warnings.filterwarnings('ignore') # Ignore all the warnings 

# Path hidpy
sys.path.append('%s/../' % os.getcwd())

# Internal packages 
import core 
from core import file_utils
from core import optical_flow
from core import video_processing
from core import plotting
from core import msd
from core import inference


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

    arg_help = 'Number of iterations, default 8'
    parser.add_argument('--iterations', help=arg_help, type=int, default=8)

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

    if args.config_file == 'EMPTY':

        video_sequence = args.input_sequence
        output_directory = args.output_directory
        pixel_threshold = args.pixel_threshold
        pixel_size = args.pixel_size

        dt = args.dt
        models_selected = list()
        if args.d_model: 
            models_selected.append('D')
        if args.da_model: 
            models_selected.append('DA')
        if args.v_model: 
            models_selected.append('V')
        if args.dv_model: 
            models_selected.append('DV')
        if args.dav_model: 
            models_selected.append('DAV')
    else:
        import configparser
        config_file = configparser.ConfigParser()

        # READ CONFIG FILE
        config_file.read(args.config_file)

        video_sequence = str(config_file['HID_PARAMETERS']['video_sequence'])
        output_directory = str(config_file['HID_PARAMETERS']['output_directory'])
        pixel_threshold = float(config_file['HID_PARAMETERS']['pixel_threshold'])
        pixel_size = float(config_file['HID_PARAMETERS']['pixel_size'])
        dt = float(config_file['HID_PARAMETERS']['dt'])
        ncores = int(config_file['HID_PARAMETERS']['n_cores'])
        
        d_model = config_file['HID_PARAMETERS']['d_model']
        da_model = config_file['HID_PARAMETERS']['da_model']
        v_model = config_file['HID_PARAMETERS']['v_model']
        dv_model = config_file['HID_PARAMETERS']['dv_model']
        dav_model = config_file['HID_PARAMETERS']['dav_model']
        
        models_selected = list()
        if d_model == 'Yes':
            models_selected.append('D')
        if da_model: 
            models_selected.append('DA')
        if v_model: 
            models_selected.append('V')
        if dv_model: 
            models_selected.append('DV')
        if dav_model: 
            models_selected.append('DAV')

    print(video_sequence)

    # Get the prefix, typically with the name of the video sequence  
    prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(video_sequence).stem, pixel_size, dt, pixel_threshold)

    ################# PLEASE DON'T EDIT THIS PANEL #################
    # Verify the input parameters, and return the path where the output data will be written  
    output_directory = file_utils.veryify_input_options(
        video_sequence=video_sequence, output_directory=output_directory, 
        pixel_threshold=pixel_threshold, pixel_size=pixel_size, dt=dt)

    # Load the frames from the video 
    frames = video_processing.get_frames_list_from_video(
        video_path=video_sequence, verbose=True)

    # Plot the first frames
    plotting.verify_plotting_packages()
    plotting.plot_frame(frame=frames[0], output_directory=output_directory, 
        frame_prefix=prefix, font_size=14, tick_count=3)


    # Compute the optical flow
    print('* Computing optical flow') 
    u, v = optical_flow.compute_optical_flow_farneback(frames=frames)


    # Interpolate the flow field
    print('* Computing interpolations')
    u, v = optical_flow.interpolate_flow_fields(u_arrays=u, v_arrays=v)


    # Compute the trajectories 
    print('* Creating trajectories')
    trajectories = optical_flow.compute_trajectories(
        frame=frames[0], fu_arrays=u, fv_arrays=v, pixel_threshold=pixel_threshold)
        
    # Plot the trajectories 
    print('* Plotting trajectories')
    trajectory_image_prefix = '%s_trajectory' % prefix
    plotting.plot_trajectories_on_frame(
        frame=frames[0], trajectories=trajectories, 
        output_path='%s/%s' % (output_directory, trajectory_image_prefix))

    # Construct trajectory map
    print('* Converting the trajectories to maps')
    xp, yp = msd.convert_trajectories_to_map(trajectories, (len(frames), frames[0].shape[0], frames[0].shape[1]))

    # Convert displacement values to microns
    xp_um = xp * pixel_size
    yp_um = yp * pixel_size

    # Extract nucleoli mask
    print('* Extracting the nucleoli mask')
    mask_nucleoli = msd.extract_nucleoli_map(xp_um, yp_um)

    # Compute the MSDs
    print('* Computing the MSDs')
    msd_array = msd.calculate_msd_for_every_pixel(xp_um, yp_um, mask_nucleoli)

    # Compute the inference, Baysian fit on MSDs 
    print('* Fitting the MSDs models using Bayesian inference')
    warnings.filterwarnings('ignore') # Ignore all the warnings 
    bayes = inference.apply_bayesian_inference(msd_array, dt, models_selected, args.n_cores)

    # The matrix that contains the mask of the nucli
    print('* Creating the maps')
    # TODO: What is the hard-coded value of 100?
    mask_matrix = numpy.zeros((frames[0].shape[0], frames[0].shape[1]))
    mask_matrix[numpy.where(mask_nucleoli == 1) ] = 100

    # Get the diffusion constant map (D)
    diffusion_constant_matrix = bayes['D']
    diffusion_constant_matrix[numpy.where(bayes['model'] == 0)] = numpy.nan
    diffusion_constant_matrix[numpy.where(bayes['D'] < 1e-10)] = numpy.nan

    # Get the anomalous exponent matrix (A)
    anomalous_exponent_matrx = bayes['A']
    anomalous_exponent_matrx[numpy.where(bayes['model'] == 0)] = numpy.nan
    anomalous_exponent_matrx[numpy.where(bayes['A'] < 1e-10)] = numpy.nan

    # Get the drift velocity matrix (V)
    drift_velocity_matrix = bayes['V']
    drift_velocity_matrix[numpy.where(bayes['model'] == 0)] = numpy.nan
    drift_velocity_matrix[numpy.where(bayes['V']==0)] = numpy.nan

    # Plot the model selection image 
    model_selection_image_prefix = '%s_model_selection' % prefix
    core.plotting.plot_model_selection_image(
        model_selection_matrix=bayes['model'], mask_matrix=mask_matrix, 
        output_directory=output_directory, frame_prefix=model_selection_image_prefix, 
        font_size=14, title='Model Selection', tick_count=3)

    # Plot the diffusion constant matrix
    d_map_image_prefix = '%s_diffusion_constant_matrix' % prefix
    core.plotting.plot_matrix_map(
        matrix=diffusion_constant_matrix, mask_matrix=mask_matrix, 
        output_directory=output_directory, frame_prefix=d_map_image_prefix, 
        font_size=14, title=r'Diffusion Constant ($\mu$m$^2$/s)', tick_count=3)

    # Plot the anomalous matrix
    a_map_image_prefix = '%s_anomalous_matrix' % prefix
    core.plotting.plot_matrix_map(
        matrix=anomalous_exponent_matrx, mask_matrix=mask_matrix, 
        output_directory=output_directory, frame_prefix=a_map_image_prefix, 
        font_size=14, title='Anomalous Exponent', tick_count=3)

    # Plot the drift velocity matrix
    v_map_image_prefix = '%s_drift_velocity_matrix' % prefix
    core.plotting.plot_matrix_map(
        matrix=drift_velocity_matrix, mask_matrix=mask_matrix, 
        output_directory=output_directory, frame_prefix=v_map_image_prefix, 
        font_size=14, title=r'Drift Velocity ($\mu$m/s)', tick_count=3)
    
    # Save pickle file per cell
    print('* Saving to picke files')
    # Create the pickle directory 
    pickle_directory = '%s/pickle' % output_directory
    file_utils.create_directory(pickle_directory)
    with open('%s/%s.pickle' % (pickle_directory, prefix), 'wb') as f:
        pickle.dump(bayes, f)

    # Generate report
    print('* Creating reports') 
    file_utils.create_report_1_summary(output_directory=output_directory,
                                       frame_0=prefix,
                                       trajectory=trajectory_image_prefix,
                                       model_selection=model_selection_image_prefix, 
                                       d_map=d_map_image_prefix, 
                                       a_map=a_map_image_prefix,
                                       v_map=v_map_image_prefix)
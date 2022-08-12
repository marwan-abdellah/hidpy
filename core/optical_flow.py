import scipy
import numpy 
from tqdm import tqdm
from .horn_schunck import HornSchunck


####################################################################################################
# @interpolate_flow_fields
####################################################################################################
def interpolate_flow_fields(u_arrays, v_arrays):

    # Arrays of interpolated fields 
    fu_arrays = list()
    fv_arrays = list()

    # Create the axes of the mesh grid 
    x_axis = numpy.arange(u_arrays[0].shape[0])
    y_axis = numpy.arange(u_arrays[0].shape[1])

    # For every time step 
    for t in tqdm(range(len(u_arrays)), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        
        # Interpolate 
        fu = scipy.interpolate.interp2d(x_axis, y_axis, u_arrays[t], kind='cubic')
        fv = scipy.interpolate.interp2d(x_axis, y_axis, v_arrays[t], kind='cubic')

        # Append the interpolate fields 
        fu_arrays.append(fu)
        fv_arrays.append(fv)
    
    # Return the interpolated flow fields 
    return fu_arrays, fv_arrays


####################################################################################################
# @compute_optical_flow_horn_schunck
####################################################################################################
def compute_optical_flow_horn_schunck(frames):
    
    # Displacement map arrays  
    u_arrays = list()
    v_arrays = list()

    # Each flow map is computed from two frames 
    for i in tqdm(range(len(frames) - 1), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):

        # Run the optical flow method
        V, U = HornSchunck(frames[i], frames[i + 1], alpha=0.001, iterations=1)

        u_arrays.append(U)
        v_arrays.append(V)

    # Return the Displacement maps 
    return u_arrays, v_arrays


####################################################################################################
# @compute_trajectory
####################################################################################################
def compute_trajectory(x0, y0, fu_arrays, fv_arrays):

    # A list containing the trajectories 
    trajectory = list()

    # Initially, the current pixel is the seed where the trajectory is starting 
    x_current = x0 
    y_current = y0

    # For every time-frame 
    for t in range(len(fu_arrays)):
        
        # Compute the interpolated displacements 
        dx = fu_arrays[t](x_current, y_current)
        dy = fv_arrays[t](x_current, y_current)

        # Compute the new coordinates 
        x_new = x_current + dx
        y_new = y_current + dy

        # Add the x_pixel and y_pixel to the list 
        trajectory.append([x_new, y_new])

        x_current = (x_new)
        y_current = (y_new)

    # Return a reference to the trajectory list 
    return trajectory


####################################################################################################
# @compute_trajectories
####################################################################################################
def compute_trajectories(frame, fu_arrays, fv_arrays, pixel_threshold=15):

    # A list containing all the trajectories 
    trajectories = list()
    for ii in tqdm(range(frame.shape[0]), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        for jj in range(frame.shape[1]):
            if frame[ii, jj] > pixel_threshold:
                trajectories.append(compute_trajectory(ii, jj, fu_arrays, fv_arrays))
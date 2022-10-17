import numpy 


####################################################################################################
# @extract_nucleoli_map
####################################################################################################
def extract_nucleoli_map(xp, yp):    
    return numpy.all(numpy.logical_and(xp==0, yp==0), axis=0)


####################################################################################################
# @convert_trajectories_to_map
####################################################################################################
def convert_trajectories_to_map(trajectories, frame_size):    
    
    xp = numpy.zeros(frame_size)
    yp = numpy.zeros(frame_size)

    for trajectory in trajectories:
        
        # first data point
        ix, iy = int(trajectory[0][0]), int(trajectory[0][1])
            
        # the first point is not an array, let's make it one
        trajectory[0] = [numpy.array([trajectory[0][0]]), numpy.array([trajectory[0][1]])]

        # add trajectory to xp and yp
        xp[:, ix, iy] = [x[0][0] for x in trajectory]
        yp[:, ix, iy] = [x[1][0] for x in trajectory]
        
    return xp, yp


####################################################################################################
# @calculate_msd_for_every_pixel
####################################################################################################
def calculate_msd_for_every_pixel(xp, yp, mask):
    """
    MSDcalculation: calculates MSD for every pixel.

    :param xp: x-position of every pixel for time t 
    :param yp: y-position of every pixel for time t 
    :param mask: mask with 0 outside nucleus and 1 inside nucleoli
    :return: MSD curve at every pixel

    """

    frame_size = len(xp)
    mask[mask == 0] = numpy.nan
    
    # t = np.arange(dT,(framesize+1)*dT,dT) # not used

    msd_list = numpy.zeros(((frame_size-1),xp.shape[1],xp.shape[2]))

    for lag in range(1,frame_size):
        d = numpy.square(xp[(lag):] - xp[0:(frame_size-lag)]) + numpy.square(yp[(lag):] - yp[0:((frame_size)-lag)]) 
        d[d==0] = numpy.nan
        msd_list[lag-1] = numpy.nanmean(d, axis=0)*mask
    
    return msd_list
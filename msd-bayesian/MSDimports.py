import numpy as np

def extract_nucleoli_map(xp, yp):    
    return np.all(np.logical_and(xp==0, yp==0), axis=0)

def convert_trajectories_to_map(trajectories, frameSize):    
    xp = np.zeros(frameSize)
    yp = np.zeros(frameSize)
    for trajectory in trajectories:
        # first data point
        ix, iy = int(trajectory[0][0]), int(trajectory[0][1])
            
        # the first point is not an array, let's make it one
        trajectory[0] = [np.array([trajectory[0][0]]), np.array([trajectory[0][1]])]

        # add trajectory to xp and yp
        xp[:,ix,iy] = [x[0][0] for x in trajectory]
        yp[:,ix,iy] = [x[1][0] for x in trajectory]
    return xp, yp

def MSDcalculation(xp, yp, mask):
    """
    MSDcalculation: calculates MSD for every pixel.

    :param xp: x-position of every pixel for time t 
    :param yp: y-position of every pixel for time t 
    :param mask: mask with 0 outside nucleus and 1 inside nucleoli
    :return: MSD curve at every pixel

    """

    framesize = len(xp)
    mask[mask == 0] = np.nan
    
    # t = np.arange(dT,(framesize+1)*dT,dT) # not used

    MSD = np.zeros(((framesize-1),xp.shape[1],xp.shape[2]))

    for lag in range(1,framesize):
        d = np.square(xp[(lag):] - xp[0:(framesize-lag)]) + np.square(yp[(lag):] - yp[0:((framesize)-lag)]) 
        d[d==0] = np.nan
        MSD[lag-1] = np.nanmean(d, axis=0)*mask
    
    return MSD



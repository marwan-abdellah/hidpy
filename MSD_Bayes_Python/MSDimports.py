import numpy as np


def MSDcalculation(xp, yp, mask, dT):
    """
    MSDcalculation: calculates MSD for every pixel.

    :param xp: x-position of every pixel for time t 
    :param yp: y-position of every pixel for time t 
    :param mask: mask with 0 outside nucleus and 1 inside nucleoli
    :param dT: time lag in sec
    :return: MSD curve at every pixel

    """

    framesize=len(xp)
    mask[mask == 0] = np.nan
    
    # t = np.arange(dT,(framesize+1)*dT,dT) # not used

    MSD = np.zeros(((framesize-1),xp.shape[1],xp.shape[2]))

    for lag in range(1,framesize):
        d = np.square(xp[(lag):] - xp[0:(framesize-lag)]) + np.square(yp[(lag):] - yp[0:((framesize)-lag)]) 
        d[d==0] = np.nan
        MSD[lag-1] = np.nanmean(d, axis=0)*mask
    
    return MSD



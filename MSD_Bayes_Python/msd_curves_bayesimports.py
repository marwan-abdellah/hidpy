import numpy as np
from MSD_Bayes_Python import msd_fittingimports


def cov_shrinkage(M,target): 
    S=[]   

    # Unbiased empirical covariance matrix 
    S=np.cov(M, rowvar=False)

        # Target matrix
    if target == 1:
        T = np.eye(S.shape[0],S.shape[1]) * np.mean(np.diag(S))
    elif target == 2:
        T = np.diag(np.diag(S))

    n = M.shape[0] #number of observations

    # Variance in covariance matrix elements (for calculating weighting factor below)

    varS = np.zeros(S.shape)

    for i in np.arange(1,S.shape[0]+1):
        for j in np.arange(1,S.shape[0]+1):
            w = np.multiply(M[:,i-1],M[:,j-1])
            varS[i-1,j-1] = (n / ((n - 1) ** 3))* sum((w - np.mean(w)) ** 2)

    # Weighting factor
    if target == 1:
        lambda_ = np.sum(varS) / np.sum((S - T) ** 2)
    elif target == 2:
        lambda_ = np.sum(varS - np.diag(np.diag(varS))) / np.sum((S - np.diag(np.diag(S)) )** 2)
        
    # Inferred covariance matrix
    Sstar = lambda_ * T + (1 - lambda_) * S
    return Sstar

    
def msd_curves_bayes(timelags,MSD_curves,msd_params): 
    results={}
   
    #### Mean MSD curve ####
    MSD_mean = np.mean(MSD_curves,1)
    MSD_mean_se = np.std(MSD_curves,1,ddof=1)/np.sqrt(MSD_curves.shape[1])

    #### Covariance matrix ####
    errors=np.zeros(MSD_curves.shape)

    # Get difference between each individual curve and the mean curve
    for j in np.arange(1,MSD_curves.shape[1]+1):
            errors[:,j-1] = MSD_curves[:,j-1] - MSD_mean

    errors = np.transpose(errors)

    # Calculate raw covariance matrix
    msd_params_error_cov_raw= np.cov(errors, rowvar=False)


    msd_params_error_cov=cov_shrinkage(errors,1)   

    # Covariance of the mean curve
    msd_params_error_cov_raw = msd_params_error_cov_raw / errors.shape[0]
    msd_params_error_cov = msd_params_error_cov / errors.shape[0]

    msd_params['error_cov_raw']=msd_params_error_cov_raw
    msd_params['error_cov']=msd_params_error_cov
    
    results['mean_curve']=msd_fittingimports.msd_fitting(timelags,MSD_mean,msd_params)

    results['timelags'] = timelags
    results['mean_curve']['MSD_vs_timelag'] = MSD_mean
    results['mean_curve']['MSD_vs_timelag_se'] = MSD_mean_se
    results['msd_params'] = results['mean_curve']['msd_params']
    results['MSD_vs_timelag'] = MSD_curves
    
    return results


    
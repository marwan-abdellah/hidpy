import numpy as np
from tqdm import tqdm

import numpy.matlib
import multiprocessing

from joblib import Parallel, delayed

import numpy as np
from scipy.optimize import least_squares
from scipy.stats.distributions import t


def lsqcurvefit_GLS(funcstr,b0,x,y,error_cov,lowBound,upBound): 
    """
    lsqcurvefit_GLS: Performs generalized least squares using the least_squares function of scipy.optimize.
    Requires an error covariance matrix error_cov for the data y. 

    :param funcstr: function which computes the vector of residuals
    :param b0: initial guess on independent variables
    :param x: xdata 
    :param y: ydata
    :param error_cov: error covariance matrix
    :param lowBound: lower bound
    :param upBound: upper bound
    :return: dictionary containing the fields: model, Se(MSD offset), D(Diffusion constant), A(Anomalous exponennt) and V (Velocity)

    """
    global L0

    L0=[]

    try:
        L0=np.linalg.cholesky(error_cov)
    except:
#       print('Warning: regularization failed')
        pass
    
    exec('global f_N0 \ndef f_N0(x,b): return ' + funcstr)
    exec('global f_N \ndef f_N(x,b): return np.linalg.solve(L0,' + funcstr+')')
  
    def f_Nerr(b,x, y):
        return (f_N(x,b)-y) 

    res_lsq= least_squares(f_Nerr, b0, args=(x,np.linalg.solve(L0,y)),bounds=(lowBound, upBound))
    
    coeffs_N=res_lsq.x
    J=res_lsq.jac
    residuals_N = y - f_N0(x,coeffs_N)
    resnorm_N = np.sum(residuals_N**2)
    
    return coeffs_N, resnorm_N, residuals_N, J, L0
    
def msd_fitting(timelags,MSD_vs_timelag,msd_params): 
    """
    msd_fitting: Fits the provided MSD curve with the models contained in msd_params['models']

    :param timelags: time vector
    :param MSD_vs_timelag: Mean MSD curve
    :param msd_params: ['models'] model selection by labels and ['error_conv'] error covariance matrix
    :return: dictionary containing the fields: model, Se(MSD offset), D(Diffusion constant), A(Anomalous exponennt) and V (Velocity)

    """

    # dictionary results
    results = {}

    msd_params['prior_scale'] = 200
    msd_params['lower_b']={}
    msd_params['upper_b']={}
    msd_params['lower_b']['C'] = 0
    msd_params['upper_b']['C'] = 1000
    msd_params['lower_b']['D'] = 0
    msd_params['upper_b']['D'] = 1000
    msd_params['lower_b']['A'] = 0
    msd_params['upper_b']['A'] = 1
    msd_params['lower_b']['R'] = 0
    msd_params['upper_b']['R'] = 1000
    msd_params['lower_b']['V'] = 0
    msd_params['upper_b']['V'] = 1000
    msd_params['lower_b']['E'] = 0
    msd_params['upper_b']['E'] = 100

    results['msd_params'] = msd_params

    ###### FITTING #######
        
    nanvalues = np.where(np.isnan(MSD_vs_timelag))
    MSD_vs_timelag = np.delete(MSD_vs_timelag, nanvalues)

    timelags = np.delete(timelags, nanvalues)
    
    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    dof=np.zeros(3)
    dof[0]= max(0, len(MSD_vs_timelag) - 1) # number of degrees of freedom for 1 parameter
    dof[1]= max(0, len(MSD_vs_timelag) - 2) # number of degrees of freedom for 2 parameters
    dof[2]= max(0, len(MSD_vs_timelag) - 3) # number of degrees of freedom for 3 parameters    
    tval = t.ppf(1.0-alpha/2., dof) # student-t value for the dof and confidence level
    
    if any('N' in s for s in msd_params['models']):
    # Fit MSD with a constant: MSD = C = 6*sigmaE^2
        J=[]
        L=[]
    
        nparam_N = 1
        b0=0.01
        funcstr='b-x+x'

        coeffs_N, resnorm_N, residuals_N, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'],msd_params['lower_b']['C'], msd_params['upper_b']['C'])
        COVB_N = resnorm_N/(len(MSD_vs_timelag)-1)*np.linalg.inv(np.dot(np.transpose(J),J))
        
        se = np.sqrt(np.diag(np.abs(COVB_N)))
        delta = se * tval[nparam_N-1]

        results['N'] = {}
        results['N'] ={'C': coeffs_N ,'C_se': se ,'C_cilo': coeffs_N-delta , 'C_cihi': coeffs_N+delta}
    
    if any('D' in s for s in msd_params['models']):
    # Fit MSD with diffusion alone: MSD = 6*D*t
        J=[]
        L=[]

        nparam_D = 1
        b0=0.01
        funcstr='6*b*x'

        coeffs_D, resnorm_D, residuals_D, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'], msd_params['lower_b']['D'], msd_params['upper_b']['D'])
        COVB_D = resnorm_D/(len(MSD_vs_timelag)-1)*np.linalg.inv(np.dot(np.transpose(J),J))
        
        se = np.sqrt(np.diag(np.abs(COVB_D)))
        delta = se * tval[nparam_D-1]

        results['D'] = {}
        results['D'] = {'D': coeffs_D ,'D_se': se ,'D_cilo': coeffs_D-delta , 'D_cihi': coeffs_D+delta}

    if any('DA' in s for s in msd_params['models']):
    # Fit MSD with anomalous diffusion alone: MSD = 6*D*t^alpha
        J=[]
        L=[]

        nparam_DA = 2
        b0=[0.01,1]
        funcstr='6*b[0]*x**(b[1])'

        coeffs_DA, resnorm_DA, residuals_DA, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'],[msd_params['lower_b']['D'],msd_params['lower_b']['A']],[msd_params['upper_b']['D'],msd_params['upper_b']['A']])  
        COVB_DA = resnorm_DA/(len(MSD_vs_timelag)-1)*np.linalg.inv(np.dot(np.transpose(J),J))
                
        se = np.sqrt(np.diag(np.abs(COVB_DA)))       
        delta = se * tval[nparam_DA-1]
        
        results['DA'] = {}
        results['DA'] = {'D': coeffs_DA[0] ,'D_se': se[0] ,'D_cilo': coeffs_DA[0]-delta[0] , 'D_cihi': coeffs_DA[0]+delta[0],'A': coeffs_DA[1] ,'A_se': se[1] ,'A_cilo': coeffs_DA[1]-delta[1] , 'A_cihi': coeffs_DA[1]+delta[1]}

    if any('DR' in s for s in msd_params['models']):
    # Fit MSD for diffusion in a reflective sphere without flow: MSD = R^2*(1-exp(-6*D*t/R^2))
        J=[]
        L=[]
        
        nparam_DR = 2
        b0=[0.01,1]
        funcstr='(b[1]**2)*(1 - np.exp(-6*b[0]*x/(b[1]**2)))'

        coeffs_DR, resnorm_DR, residuals_DR, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'],[msd_params['lower_b']['D'],msd_params['lower_b']['R']],[msd_params['upper_b']['D'],msd_params['upper_b']['R']])  
        COVB_DR = resnorm_DR/(len(MSD_vs_timelag)-1)*np.linalg.inv(np.dot(np.transpose(J),J))
                
        se = np.sqrt(np.diag(np.abs(COVB_DR)))       
        delta = se * tval[nparam_DR-1]
                
        results['DR'] = {}
        results['DR'] = {'D': coeffs_DR[0] ,'D_se': se[0] ,'D_cilo': coeffs_DR[0]-delta[0] , 'D_cihi': coeffs_DR[0]+delta[0],'R': coeffs_DR[1] ,'R_se': se[1] ,'R_cilo': coeffs_DR[1]-delta[1] , 'R_cihi': coeffs_DR[1]+delta[1]}

    if any('V' in s for s in msd_params['models']):
    # Fit MSD with flow alone: MSD = (v*t)^2
        J=[]
        L=[]

        nparam_V = 1
        b0=0.01
        funcstr='(b**2)*(x**2)'

        coeffs_V, resnorm_V, residuals_V, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'], msd_params['lower_b']['V'], msd_params['upper_b']['V'])
        mat1=np.linalg.inv(np.dot(np.transpose(J),J))
        COVB_V = resnorm_V/(len(MSD_vs_timelag)-1)*mat1
        
        se = np.sqrt(np.diag(np.abs(COVB_V)))
        delta = se * tval[nparam_V-1]

        results['V'] = {}
        results['V'] = {'V': coeffs_V ,'V_se': se ,'V_cilo': coeffs_V-delta , 'V_cihi': coeffs_V+delta}

    if any('DV' in s for s in msd_params['models']):
    # Fit MSD with diffusion plus flow: MSD = 6*D*t + (v*t)^2
        J=[]
        L=[]

        nparam_DV = 2
        b0=[0.01,0.01]
        funcstr='b[0]*(6*x)+(b[1]**2)*(x**2)'

        coeffs_DV, resnorm_DV, residuals_DV, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'],[msd_params['lower_b']['D'],msd_params['lower_b']['V']],[msd_params['upper_b']['D'],msd_params['upper_b']['V']])  
        COVB_DV = resnorm_DV/(len(MSD_vs_timelag)-1)*np.linalg.inv(np.dot(np.transpose(J),J))
                
        se = np.sqrt(np.diag(np.abs(COVB_DV)))       
        delta = se * tval[nparam_DV-1]
        
        results['DV'] = {}
        results['DV'] = {'D': coeffs_DV[0] ,'D_se': se[0] ,'D_cilo': coeffs_DV[0]-delta[0] , 'D_cihi': coeffs_DV[0]+delta[0],'V': coeffs_DV[1] ,'V_se': se[1] ,'V_cilo': coeffs_DV[1]-delta[1] , 'V_cihi': coeffs_DV[1]+delta[1]}

    if any('DAV' in s for s in msd_params['models']):
    # Fit MSD with anomalous diffusion plus flow: MSD = 6*D*t^alpha + (v*t)^2
        J=[]
        L=[]

        nparam_DAV = 3
        b0=[0.01, 1, 0.01]
        funcstr='6*b[0]*(x**b[1])+(b[2]**2)*(x**2)'

        coeffs_DAV, resnorm_DAV, residuals_DAV, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'],[msd_params['lower_b']['D'],msd_params['lower_b']['A'],msd_params['lower_b']['V']],[msd_params['upper_b']['D'],msd_params['upper_b']['A'],msd_params['upper_b']['V']])  
        COVB_DAV = resnorm_DAV/(len(MSD_vs_timelag)-1)*np.linalg.inv(np.dot(np.transpose(J),J))
        
        se = np.sqrt(np.diag(np.abs(COVB_DAV)))       
        delta = se * tval[nparam_DAV-1]
        
        results['DAV'] = {}
        results['DAV'] = {'D': coeffs_DAV[0] ,'D_se': se[0] ,'D_cilo': coeffs_DAV[0]-delta[0] , 'D_cihi': coeffs_DAV[0]+delta[0],'A': coeffs_DAV[1] ,'A_se': se[1] ,'A_cilo': coeffs_DAV[1]-delta[1] , 'A_cihi': coeffs_DAV[1]+delta[1],'V': coeffs_DAV[2] ,'V_se': se[2] ,'V_cilo': coeffs_DAV[2]-delta[2] , 'V_cihi': coeffs_DAV[2]+delta[2]}
      
    if any('DRV' in s for s in msd_params['models']):
    # Fit MSD for diffusion in a reflective sphere plus flow: MSD = R^2*(1-exp(-6*D*t/R^2)) + (v*t)^2
        J=[]
        L=[]

        nparam_DRV = 3
        b0=[0.01, 1, 0.01]
        funcstr='(b[1]**2)*(1-np.exp(-6*b[0]*x/(b[1]**2)))+(b[2]**2)*(x**2)'

        coeffs_DRV, resnorm_DRV, residuals_DRV, J,L=lsqcurvefit_GLS(funcstr,b0,timelags,MSD_vs_timelag,msd_params['error_cov'],[msd_params['lower_b']['D'],msd_params['lower_b']['R'],msd_params['lower_b']['V']],[msd_params['upper_b']['D'],msd_params['upper_b']['R'],msd_params['upper_b']['V']])  
        mat1=np.linalg.inv(np.dot(np.transpose(J),J))
        COVB_DRV = resnorm_DRV/(len(MSD_vs_timelag)-1)*mat1

        se = np.sqrt(np.diag(np.abs(COVB_DRV)))       
        delta = se * tval[nparam_DRV-1]
        
        results['DRV'] = {}
        results['DRV'] = {'D': coeffs_DRV[0] ,'D_se': se[0] ,'D_cilo': coeffs_DRV[0]-delta[0] , 'D_cihi': coeffs_DRV[0]+delta[0],'R': coeffs_DRV[1] ,'R_se': se[1] ,'R_cilo': coeffs_DRV[1]-delta[1] , 'R_cihi': coeffs_DRV[1]+delta[1],'V': coeffs_DRV[2] ,'V_se': se[2] ,'V_cilo': coeffs_DRV[2]-delta[2] , 'V_cihi': coeffs_DRV[2]+delta[2]}

    ### Likelihoods and model probabilities ###
    
    MLsum=0 #normalization factor for model probabilities

    # Calculate the log probability for each model

    if any('N' in s for s in msd_params['models']):
        results_N_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_N,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_N))+ 0.5*nparam_N*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_N)))
        results_N_logML = results_N_logML - np.log(2*(results['N']['C_se'])*msd_params['prior_scale'])

        if np.isnan(results_N_logML):
            results_N_logML = np.NINF

        results['N']['logML']=results_N_logML

        MLsum = MLsum + np.exp(results_N_logML)
 
    if any('D' in s for s in msd_params['models']):
        results_D_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_D,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_D))+ 0.5*nparam_D*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_D)))
        results_D_logML = results_D_logML - np.log(2*(results['D']['D_se'])*msd_params['prior_scale'])

        if np.isnan(results_D_logML):
            results_D_logML = np.NINF

        results['D']['logML']=results_D_logML

        MLsum = MLsum + np.exp(results_D_logML)
       
    if any('DA' in s for s in msd_params['models']):
        results_DA_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_DA,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_DA))+ 0.5*nparam_DA*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_DA)))
        results_DA_logML = results_DA_logML - np.log(2*(results['DA']['D_se'])*msd_params['prior_scale']*2*(results['DA']['A_se'])*msd_params['prior_scale'])

        if np.isnan(results_DA_logML):
            results_DA_logML = np.NINF

        results['DA']['logML']=results_DA_logML

        MLsum = MLsum + np.exp(results_DA_logML)   

    if any('DR' in s for s in msd_params['models']):
        results_DR_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_DR,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_DR))+ 0.5*nparam_DR*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_DR)))
        results_DR_logML = results_DR_logML - np.log(2*(results['DR']['D_se'])*msd_params['prior_scale']*2*(results['DR']['R_se'])*msd_params['prior_scale'])

        if np.isnan(results_DR_logML):
            results_DR_logML = np.NINF

        results['DR']['logML']=results_DR_logML

        MLsum = MLsum + np.exp(results_DR_logML)

    if any('V' in s for s in msd_params['models']):  
        results_V_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_V,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_V))+ 0.5*nparam_V*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_V)))
        results_V_logML = results_V_logML - np.log(2*(results['V']['V_se'])*msd_params['prior_scale'])

        if np.isnan(results_V_logML):
            results_V_logML = np.NINF
        
        results['V']['logML']=results_V_logML

        MLsum = MLsum + np.exp(results_V_logML)

    if any('DV' in s for s in msd_params['models']):  
        results_DV_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_DV,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_DV))+ 0.5*nparam_DV*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_DV)))
        results_DV_logML = results_DV_logML - np.log(2*(results['DV']['D_se'])*msd_params['prior_scale']*2*(results['DV']['V_se'])*msd_params['prior_scale'])

        if np.isnan(results_DV_logML):
            results_DV_logML = np.NINF
        
        results['DV']['logML']=results_DV_logML

        MLsum = MLsum + np.exp(results_DV_logML)

    if any('DAV' in s for s in msd_params['models']):
        results_DAV_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_DAV,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_DAV))+ 0.5*nparam_DAV*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_DAV)))
        results_DAV_logML = results_DAV_logML - np.log(2*(results['DAV']['D_se'])*msd_params['prior_scale']*2*(results['DAV']['A_se'])*msd_params['prior_scale']*2*(results['DAV']['V_se'])*msd_params['prior_scale'])

        if np.isnan(results_DAV_logML):
            results_DAV_logML = np.NINF
        
        results['DAV']['logML']=results_DAV_logML

        MLsum = MLsum + np.exp(results_DAV_logML) 

    if any('DRV' in s for s in msd_params['models']):
        results_DRV_logML = - 0.5 * np.sum(np.dot(np.dot(np.dot(residuals_DRV,np.transpose(np.linalg.inv(L))),(np.linalg.inv(L))),residuals_DRV))+ 0.5*nparam_DRV*np.log(2*np.pi)+ 0.5*np.log(abs(np.linalg.det(COVB_DRV)))
        results_DRV_logML = results_DRV_logML - np.log(2*(results['DRV']['D_se'])*msd_params['prior_scale']*2*(results['DRV']['R_se'])*msd_params['prior_scale']*2*(results['DRV']['V_se'])*msd_params['prior_scale'])

        if np.isnan(results_DRV_logML):
            results_DRV_logML = np.NINF
        
        results['DRV']['logML']=results_DRV_logML

        MLsum = MLsum + np.exp(results_DRV_logML)

    # Calculate the normalized probability of each model
    
    if any('N' in s for s in msd_params['models']):       
        if (np.exp(results['N']['logML'])==0) and (MLsum==0):
            results['N']['PrM'] =np.nan
        elif (MLsum==0):
            results['N']['PrM'] =np.inf
        else:
            results['N']['PrM'] = np.exp(results['N']['logML'])/MLsum  

    if any('D' in s for s in msd_params['models']): 
        if (np.exp(results['D']['logML'])==0) and (MLsum==0):
            results['D']['PrM'] =np.nan
        elif (MLsum==0):
            results['D']['PrM'] =np.inf
        else:
            results['D']['PrM'] = np.exp(results['D']['logML'])/MLsum  
    

    if any('DA' in s for s in msd_params['models']): 
        if (np.exp(results['DA']['logML'])==0) and (MLsum==0):
            results['DA']['PrM'] =np.nan
        elif (MLsum==0):
            results['DA']['PrM'] =np.inf
        else:
            results['DA']['PrM'] = np.exp(results['DA']['logML'])/MLsum  
    
    if any('DR' in s for s in msd_params['models']):        
        if (np.exp(results['DR']['logML'])==0) and (MLsum==0):
            results['DR']['PrM'] =np.nan
        elif (MLsum==0):
            results['DR']['PrM'] =np.inf
        else:
            results['DR']['PrM'] = np.exp(results['DR']['logML'])/MLsum  

    if any('V' in s for s in msd_params['models']): 
        if (np.exp(results['V']['logML'])==0) and (MLsum==0):
            results['V']['PrM'] =np.nan
        elif (MLsum==0):
            results['V']['PrM'] =np.inf
        else:
            results['V']['PrM'] = np.exp(results['V']['logML'])/MLsum  

    if any('DV' in s for s in msd_params['models']): 
        if (np.exp(results['DV']['logML'])==0) and (MLsum==0):
            results['DV']['PrM'] =np.nan
        elif (MLsum==0):
            results['DV']['PrM'] =np.inf
        else:
            results['DV']['PrM'] = np.exp(results['DV']['logML'])/MLsum  

    if any('DAV' in s for s in msd_params['models']):         
        if (np.exp(results['DAV']['logML'])==0) and (MLsum==0):
            results['DAV']['PrM'] =np.nan
        elif (MLsum==0):
            results['DAV']['PrM'] =np.inf
        else:
            results['DAV']['PrM'] = np.exp(results['DAV']['logML'])/MLsum  

    if any('DRV' in s for s in msd_params['models']): 
        if (np.exp(results['DRV']['logML'])==0) and (MLsum==0):
            results['DRV']['PrM'] =np.nan
        elif (MLsum==0):
            results['DRV']['PrM'] =np.inf
        else:
            results['DRV']['PrM'] = np.exp(results['DRV']['logML'])/MLsum  

    return results

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
    
    results['mean_curve']=msd_fitting(timelags,MSD_mean,msd_params)

    results['timelags'] = timelags
    results['mean_curve']['MSD_vs_timelag'] = MSD_mean
    results['mean_curve']['MSD_vs_timelag_se'] = MSD_mean_se
    results['msd_params'] = results['mean_curve']['msd_params']
    results['MSD_vs_timelag'] = MSD_curves
    
    return results


####################################################################################################
# @apply_bayesian_inference
####################################################################################################
def apply_bayesian_inference(MSD, dT,models_selected, num_cores=0): 
    """
    MSDBayes: Applies a Bayesian inference to a small subset of up to 10
    trajectories and thereby chooses the best fitting model considering the
    model complexity and the corresponding parameters. Model selection and
    parameters are mapped onto the projected nuclear volume.

    :param MSD: MSD curve at every pixel
    :param dT: time lag in sec
    :param models_selected: model selection by labels
    :return: dictionary containing the fields: model, Se(MSD offset), D(Diffusion constant), A(Anomalous exponennt) and V (Velocity)

    """

    # Initialization outputs
    Bayes={}
    results={}

    # set models to choose from
    msd_params={}
    msd_params['models'] = models_selected
    NumModels = len(msd_params['models'])

    # time vector
    timelags=np.arange(dT,(np.round(0.5 * (len(MSD)+1))+1)*dT,dT)

    # number of trajectories to process
    amount = np.count_nonzero(~np.isnan(MSD[0]))

    # initialize output
    prob = np.zeros((MSD[0].shape[0],MSD[0].shape[1],NumModels))
    model = np.zeros(MSD[0].shape)
    Se = np.zeros(MSD[0].shape)
    D = np.zeros(MSD[0].shape)
    A = np.zeros(MSD[0].shape)
    V = np.zeros(MSD[0].shape)  
    
    #  loop through pixels
    print('Bayesian inference..')

    
    yx_coords = np.column_stack(np.where(~np.isnan(MSD[0])))
    
    if num_cores == 0:
        ncores = multiprocessing.cpu_count()
        print('Using # cores:'+str(round(ncores)))
        results = Parallel(n_jobs=round(ncores))(delayed(func)(MSD,i,yx_coords,timelags,msd_params) for i in tqdm(range(yx_coords.shape[0])))
    elif num_cores > 1:
        print('Using # cores:' + str(round(num_cores)))
        results = Parallel(n_jobs=round(num_cores))(delayed(func)(MSD,i,yx_coords,timelags,msd_params) for i in tqdm(range(yx_coords.shape[0])))
    else:
        ### No parallel
        print('Using # cores: 1')
        for i in tqdm(range(yx_coords.shape[0])):
            results[i]=func(MSD,i,yx_coords,timelags,msd_params)
    
    count=0
   
    for i in tqdm(range(len(results))):
        if not results[i]['ERROR']:
            count=count+1
            row=results[i]['row']
            col=results[i]['col']
                      
             # extract model probabilities and find maximum probability
            
            for jj in range(prob.shape[2]):
                prob[row-1,col-1,jj] = np.real(results[i]['mean_curve'][msd_params['models'][jj]]['PrM'])
            
            prob[np.where(np.isnan(prob))]=0

            I=np.argmax(prob[row-1,col-1,:])
            

            criterion1=(prob[row-1,col-1,:]) < 1e-06
            criterion2=((prob[row-1,col-1,I]) <= 0.9)*np.ones(prob[row-1,col-1,:].shape,dtype=bool)

            if np.all(np.logical_or(criterion1,criterion2)):
                I=np.nan
            
            if not np.isnan(I):
                temp=results[i]['mean_curve'][msd_params['models'][I]]
                keysList=list(temp.keys())

                if any('C' in s for s in keysList):
                    Se[row-1,col-1]=results[i]['mean_curve'][msd_params['models'][I]]['C']
                
                if any('E' in s for s in keysList):
                    Se[row-1,col-1]=results[i]['mean_curve'][msd_params['models'][I]]['E']
                
                if any('D' in s for s in keysList):
                    D[row-1,col-1]=results[i]['mean_curve'][msd_params['models'][I]]['D']

                if any('A' in s for s in keysList):
                    A[row-1,col-1]=results[i]['mean_curve'][msd_params['models'][I]]['A']

                if any('V' in s for s in keysList):
                    V[row-1,col-1]=results[i]['mean_curve'][msd_params['models'][I]]['V']
                
                # chosen model ... corrected index number, first model is equals to 1
                model[row-1,col-1]=I+1


    # create output structure
    Bayes['model'] = model
    Bayes['Se'] = Se
    Bayes['D'] = D
    Bayes['A'] = A
    Bayes['V'] = V
    
    return Bayes

    
def func(MSD,i,yx_coords,timelags,msd_params):
    nh = 1
        
    coord=yx_coords[i,:]
    row=coord[0]+1
    col=coord[1]+1
    results={}

    try:
                        
        # get 3x3 neighborhood around central pixel
        if (row-1 - nh) < 0:
            xs = row-1
        else:
            xs = row-1 - nh

        if (row-1 + nh)> (MSD[0].shape[0]-1):
            xe = row-1
        else:
            xe = row-1 + nh

        if (col-1 - nh) < 0:
            ys = col-1
        else:
            ys = col-1 - nh
                
        if (col-1 + nh) > (MSD[0].shape[1]-1):
            ye = col-1
        else:
            ye = col-1 + nh

        MSD_curves = MSD[:,xs:(xe+1),ys:(ye+1)] 

        MSD_curves=np.transpose(MSD_curves,(0,2,1))
            
        MSD_curves=np.reshape(MSD_curves,(MSD_curves.shape[0],-1))
                
        # do not take MSDs into account with NaNs
        index = np.isnan(MSD_curves).any(axis=0)
        MSD_curves = np.delete(MSD_curves, index,axis=1)

        # Take only half of the MSD curve
                
        MSD_curves = MSD_curves[0:int(np.around(0.5*(MSD_curves.shape[0]+1))),:]

        # throw error if only 1 curve is available
        testresidualvalues=np.sum(np.abs(np.transpose(np.matlib.repmat(np.mean(MSD_curves,axis=1),MSD_curves.shape[1],1))-MSD_curves))
                
        if MSD_curves.shape[1] < 2 or testresidualvalues < 1e-06:
            results['ERROR'] = True
        else:
            #results = msd_curves_bayes(timelags,MSD_curves,msd_params)
            results=msd_curves_bayes(timelags,MSD_curves,msd_params)
            results['ERROR'] = False
             

        results['row']=row
        results['col']=col
                         
        #return results,MSD_curves
        return results

    except:
       results['ERROR']=True
       results['row']=row
       results['col']=col
       
       return results


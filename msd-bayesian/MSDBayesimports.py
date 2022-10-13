import numpy as np
from tqdm import tqdm
from MSD_Bayes_Python import msd_curves_bayesimports
import numpy.matlib
import multiprocessing

from joblib import Parallel, delayed



def MSDBayes(MSD, dT,models_selected): 
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

    parallelflag=True
    yx_coords = np.column_stack(np.where(~np.isnan(MSD[0])))
    
    if parallelflag:
        ### Parallel
        num_cores = 8       #multiprocessing.cpu_count()
        print('Using # cores:'+str(round(num_cores)))
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
            results=msd_curves_bayesimports.msd_curves_bayes(timelags,MSD_curves,msd_params)
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


   


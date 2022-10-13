# applyGMMconstrained_fitout.py
#
# Takes input from MATLAB-saved file called PythonInput.csv and does a 
# General Mixture Model estimation fot the specified distribution and
# number of populations
# mean, standard devaition and relative weight of populations is retured
# and saved to a file called PythonOutput.csv, which can be read by MATLAB
#
# this file uses normal distributions with initial parameters given by kmeans,
# which is included in the pomegranate package
# -> call GeneralMixtureModel.from_samples
#
# same file as applyGMMconstrained.py, but additional output is given, namely
# the probability of each belonging to a specific subpopulation, the 
# classification label and the distributions of the subpopulaitons

from pomegranate import NormalDistribution, LogNormalDistribution, GeneralMixtureModel
import numpy as np
import pandas

##########################################################################
###############           FUNCTIONS       ################################
##########################################################################
        
def normal(x, mu=0, sigma=1):
    fun = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))
    return fun

##########################################################################
    
def lognormal(x, mu=0, sigma=1):
    fun = 1/(x*np.sqrt(2*np.pi*sigma**2)) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
    return fun

##########################################################################

def GMMconstrained(data, DistributionType, numDist):
        
    # initialize variables list
    mu = np.zeros(3)
    sigma = np.zeros(3)
    weights = np.zeros(3)
    
    ### call GMM
    # special case 1 distribution, in which mixture model doesnt make sense
    if numDist == 1:
        if DistributionType == 'normal':
            model = NormalDistribution.from_samples(data)
        elif DistributionType == 'lognormal':
            model = LogNormalDistribution.from_samples(data)
            
    else:
        if DistributionType == 'normal':
            model = GeneralMixtureModel.from_samples(NormalDistribution, numDist, data)
        elif DistributionType == 'lognormal':
            model = GeneralMixtureModel.from_samples(LogNormalDistribution, numDist, data)
      
    x = np.expand_dims( \
        np.linspace(np.max((1e-16, data.min()-np.ptp(data)*0.25)), \
                    data.max()+np.ptp(data)*0.25, 500), \
                    1)
    if numDist == 1:   
        
        weights = [1, 0, 0]
        # extract labels, which are all zero (only one distribution)
        labels = np.zeros(data.shape)
        #  probability of each data point belonging to the chosen distribution
        # is set to one since we don't have a choice        
        y_prob = np.ones(x.shape)
        # get probability distribution
        p_model = model.probability(x)
        # get probabiity distribution of populations
        p_pop = p_model
        
        if DistributionType == 'normal':
            mu[0] = model.parameters[0]
            sigma[0] = model.parameters[1]
        elif DistributionType == 'lognormal':
            # if lognormal is chosen, convert parameters to gaussian parameters
            mu[0] = np.exp( model.parameters[0] + 0.5*model.parameters[1]**2 )
            sigma[0] = np.exp( 2*model.parameters[0] + model.parameters[1]**2 ) * (np.exp(model.parameters[1]**2) - 1)
    else:        
        
        # extract labels
        labels = model.predict(data)
        #labels = model.predict_proba( data ) # "Soft' classification
        # get probability of each data point belonging to the chosen distribution
        y_prob = model.predict_proba(x)  
        # get probability distribution
        p_model = np.exp(model.log_probability(x))
        # get probabiity distribution of populations
        p_pop = np.zeros((len(x), numDist))
        
        for d in range(numDist):
            param = model.distributions[d].parameters
            weights[d] = np.exp(model.weights[d])
            if DistributionType == 'normal':
                mu[d] = param[0]
                sigma[d] = param[1]                
                p_pop[:,d] = weights[d]*normal(x[:,0], mu[d], sigma[d])
            
            elif DistributionType == 'lognormal':
                # if lognormal is chosen, convert parameters to gaussian parameters
                mu[d] = np.exp( param[0] + 0.5*param[1]**2 )
                sigma[d] = np.exp( 2*param[0] + param[1]**2 ) * (np.exp(param[1]**2) - 1)                
                p_pop[:,d] = weights[d]*lognormal(x[:,0], param[0], param[1])
              
            
    return mu, sigma, weights, labels, x, y_prob, p_model, p_pop, model
        
##########################################################################
        

def applyGMMfun(data,DistributionType,numDist):

    outmat={}

    # do GMM estimation
    mu, sigma, weights, labels, x, y_prob, p_model, p_pop, model = GMMconstrained(data, DistributionType, numDist)


    # # create array which contains mu, sigma and weights
    # outvar_param = pandas.DataFrame( np.concatenate((mu, sigma, weights)) )
    # outvar_labels = pandas.DataFrame( labels )
    # outvar_y_prob = pandas.DataFrame( y_prob )
    # outvar_p_model = pandas.DataFrame( p_model )
    # outvar_x = pandas.DataFrame( x )
    # outvar_p_pop = pandas.DataFrame( p_pop )

    # # and save to file   
    # outvar_param.to_csv(['PythonOutput_param.csv', index=False)
    # outvar_labels.to_csv('PythonOutput_labels.csv', index=False)
    # outvar_y_prob.to_csv('PythonOutput_y_prob.csv', index=False)
    # outvar_p_model.to_csv('PythonOutput_p_model.csv', index=False)
    # outvar_x.to_csv('PythonOutput_x.csv', index=False)
    # outvar_p_pop.to_csv('PythonOutput_p_pop.csv', index=False)

    outmat['mu']=mu
    outmat['sigma']=sigma
    outmat['weights']=weights
    outmat['labels']=labels
    outmat['y_prob']=y_prob
    outmat['p_model']=p_model
    outmat['x']=x
    outmat['p_pop']=p_pop
    outmat['model']=model


    return outmat
        

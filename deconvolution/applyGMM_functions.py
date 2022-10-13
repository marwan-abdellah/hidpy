# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:57:13 2018

@author: Roman
"""
# applyGMM.py
#
# Takes input from MATLAB-saved file called PythonInput.csv and does a 
# General Mixture Model estimation for normal and lognormal probability 
# density function (each for 1-3 popualtions) and finds the model which 
# fits best to data among the 6 tested models. Decision criterion is BIC.
#
# the type of distribution (normal or log-normal), number of populations, 
# mean, standard devaition and relative weight of populations is retured
# and saved to a file called PythonOutput.csv, which can be read by MATLAB
#
# this file uses normal distributions with initial parameters given by kmeans,
# which is included in the pomegranate package
# -> call GeneralMixtureModel.from_samples

from pomegranate import NormalDistribution, LogNormalDistribution, GeneralMixtureModel
from pomegranate import *
import numpy as np
#from scipy.stats import chisquare
import pandas

##########################################################################
###############           FUNCTIONS       ################################
##########################################################################
def GMM(data, numDist=3):
        
    # initialize variables list
    model_normal = [None]*numDist
    model_lognormal = [None]*numDist
    logL = np.zeros((2, numDist))
    BIC = np.zeros((2, numDist))
    weights = np.zeros((2, numDist))
    
    ### call GMM
    for nd in range(numDist):
        # special case 1 distribution, in which mixture model doesnt make sense
        if nd == 0:
            model_normal[nd] = NormalDistribution.from_samples(data)
            model_lognormal[nd] = LogNormalDistribution.from_samples(data)
    
        else:
            # normal distribution
            model_normal[nd] = GeneralMixtureModel.from_samples(NormalDistribution, nd+1, data)
            # log-normal distribution
            model_lognormal[nd] = GeneralMixtureModel.from_samples(LogNormalDistribution, nd+1, data)
          
        # calcualte log-likelihood
        logL[0,nd] = np.sum(model_normal[nd].log_probability(data))
        logL[1,nd] = np.sum(model_lognormal[nd].log_probability(data))
        # calcualte number of free parameters in each model
        number_cov_params = nd + 1
        number_mean_params = nd + 1    
        n_parameters = number_cov_params + number_mean_params + (nd+1) - 1
        # Bayesian Information Criterion
        BIC[0,nd] = -2 * logL[0,nd] + n_parameters * np.log(data.shape[0])
        BIC[1,nd] = -2 * logL[1,nd] + n_parameters * np.log(data.shape[0])        
       
    # from BIC, get the winner
#    x = np.expand_dims( \
#        np.linspace(np.max((0, data.min()-np.ptp(data)*0.25)), \
#                    data.max()+np.ptp(data)*0.25, 500), \
#                    1)
        
    ind = np.nanargmin(BIC)
    if ind == 0:
        model = model_normal[0]        
#        # extract labels, which are all zero (only one distribution)
#        labels = np.zeros(data.shape)
#        #  probability of each data point belonging to the chosen distribution
#        # is set to one since we don't have a choice        
#        y_prob = np.ones(x.shape)
#        # get probability distribution
#        p_model = model.probability(x)
        # get relative weights, here ones
        weights = [1, 0, 0]
        number_populations = 1
    elif ind == 1:
        model = model_normal[1]
#        # extract labels
#        labels = model.predict(data)
#        # get probability of each data point belonging to the chosen distribution
#        y_prob = model.predict_proba(x)  
#        # get probability distribution
#        p_model = np.exp(model.log_probability(x))
        # get relative weights
        weights = np.append(np.exp(model.weights), 0)
        number_populations = 2
    elif ind == 2:
        model = model_normal[2]
#        labels = model.predict(data)
#        y_prob = model.predict_proba(x)  
#        p_model = np.exp(model.log_probability(x))
        weights = np.append(np.exp(model.weights), 0)
        number_populations = 3
    elif ind == 3:
        model = model_lognormal[0]
#        labels = np.zeros(data.shape)
#        y_prob = np.ones(x.shape)
#        p_model = model.probability(x)
        weights = [1, 0, 0]
        number_populations = 1
    elif ind == 4:
        model = model_lognormal[1]
#        labels = model.predict(data)
#        y_prob = model.predict_proba(x)  
#        p_model = np.exp(model.log_probability(x))
        weights = np.append(np.exp(model.weights), 0)
        number_populations = 2
    elif ind == 5:
        model = model_lognormal[2]        
#        labels = model.predict(data)
#        y_prob = model.predict_proba(x)  
#        p_model = np.exp(model.log_probability(x))
        weights = np.exp(model.weights)
        number_populations = 3
    
#    # condutct a chi-square test
#    h = np.std(data)*(4/3/len(data))**(1/5) # Silverman's rule of thumb for the bandwidth
#    bins = int(np.ceil((np.max(data)-np.min(data))/h))
#    B, X = np.histogram( data, bins=bins ,normed=True)
#    statistic, pvalue = chisquare(f_obs= B,   # Array of observed counts
#      f_exp= np.exp(model.log_probability(X[1:]-np.diff(X)/2)), # Array of expected counts  
#      ddof=bins-(2*number_populations+1))   # degrees of freedom
#    print(pvalue)
#    if pvalue < 0.1:
#        mu = np.zeros((1,3))
#        sigma = np.zeros((1,3))
#        weights = np.zeros((3))
#        number_populations = 0
#        DistributionType = 0
#    else:    
        
    if ind < 3:
        DistributionType = 'normal'
    else:
        DistributionType = 'lognormal'
        
        
    # extract parameters
    mu = np.zeros((numDist, 1))
    sigma = np.zeros((numDist, 1))    
    if number_populations==1:
        if DistributionType == 'normal':
            mu[0] = model.parameters[0]
            sigma[0] = model.parameters[1]
        elif DistributionType == 'lognormal':
            # if lognormal is chosen, convert parameters to gaussian parameters
            mu[0] = np.exp( model.parameters[0] + 0.5*model.parameters[1]**2 )
            sigma[0] = np.exp( 2*model.parameters[0] + model.parameters[1]**2 ) * (np.exp(model.parameters[1]**2) - 1)

    else:
        for d in range(number_populations):
            param = model.distributions[d].parameters
            if DistributionType == 'normal':
                mu[d] = param[0]
                sigma[d] = param[1]
            elif DistributionType == 'lognormal':
                # if lognormal is chosen, convert parameters to gaussian parameters
                mu[d] = np.exp( param[0] + 0.5*param[1]**2 )
                sigma[d] = np.exp( 2*param[0] + param[1]**2 ) * (np.exp(param[1]**2) - 1)
              
    
    return mu, sigma, weights, number_populations, DistributionType,model
#    return model, labels, x, y_prob, p_model, mu, sigma, weights, \
#        number_populations, DistributionType, logL, BIC
        
##########################################################################
        
def normal(x, mu=0, sigma=1):
    fun = 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))
    return fun

##########################################################################
    
def lognormal(x, mu=0, sigma=1):
    fun = 1/(x*np.sqrt(2*np.pi*sigma**2)) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
    return fun

##########################################################################
        
#plt.close("all")

# load csv file
#filename = 'PythonInput.csv'
#df = pandas.read_csv(filename, sep=' ')
#data = df.values

def applyGMMfun(data,numDist):
    outmat={}
#numDist = 3
##model, labels, x, y_prob, p_model, mu, sigma, weights, \
##    number_populations, DistributionType, logL, BIC = GMM(var, numDist)  
    mu, sigma, weights, number_populations, DistributionType,model = GMM(data, numDist)
    if DistributionType == 'normal':
        isnormal = 1
    else:
        isnormal = 0
    # create array which contains everything
    outvar = pandas.DataFrame( np.concatenate((np.expand_dims(isnormal,0), \
            np.expand_dims(number_populations,0), mu[:,0], sigma[:,0], weights)) )

    outmat['mu']=mu
    outmat['sigma']=sigma
    outmat['weights']=weights
    outmat['number_populations']=number_populations
    outmat['DistributionType']=DistributionType
    outmat['model']=model

##outvar.to_csv('PythonOutput.csv', index=False)
    return outvar, outmat
            
#plt.figure()
#plt.hist( var, edgecolor='c', color='c', normed=True, bins=50, alpha=0.3 )
#plt.plot(x, p_model)
#if DistributionType == 'normal':
#    for d in range(number_populations):
#        plt.plot(x, weights[d]*normal(x, mu[d], sigma[d]))
#else:
#    for d in range(number_populations):
#        plt.plot(x, weights[d]*lognormal(x, mu[d], sigma[d]))
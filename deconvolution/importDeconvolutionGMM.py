from deconvolution import applyGMM_functions
from deconvolution  import applyGMMconstrained_fitout_functions
import numpy as np
import pickle
from tqdm import tqdm
from pylab import *
import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import iqr
from matplotlib import colors

import seaborn
from matplotlib import pyplot
import matplotlib.pyplot as pyplot
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter


def sample_range(start,
                 end,
                 steps):

    # Delta
    delta = 1. * (end - start) / (steps - 1)

    # Data
    data = list()
    for i in range(steps):
        value = start + i * delta
        data.append(value)

    return data

####################################################################################################
# @verify_plotting_packages
####################################################################################################
def verify_plotting_packages():

    # Import the fonts
    font_dirs = ['/projects/hidpy/fonts']
    # font_dirs .extend([os.path.dirname(os.path.realpath(__file__)) + '/../fonts/'])
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)



def applyGMM_Multiple(listdir,parameters2decon,numDist):
    BayesMat={}
    count=0
    for filename in tqdm(listdir):
        GMM_input=[]
        outputmat={}

        BayesMat[count]=pickle.load(open(filename,'rb'))
        BayesMat[count]['filename']=filename
        Bayes1=BayesMat[count]
    

        for parameter2analyse in parameters2decon:
            HiD_parameter=Bayes1[parameter2analyse]            
            HiD_parameter[np.where(np.isnan(HiD_parameter))]=0
            index=np.where(HiD_parameter>1e-10)
            A = np.squeeze(np.asarray(HiD_parameter[index]))
            A=np.random.choice(A,2000)                                 # use to reduce the data to fit
            GMM_input = A.reshape(-1, 1)          
            outvar, outmat=applyGMM_functions.applyGMMfun(GMM_input,numDist)
            outputmat[parameter2analyse]=outmat
        BayesMat[count]['Deconvolution']=outputmat
            
        count=count+1 
    
    return BayesMat
    

####################################################################################################
# @apply_gmm_on_multiple_files
####################################################################################################
def apply_gmm_on_multiple_files(directory, parameters_to_deconvolve, number_distributions):
    
    # Build this dictionary 
    bayes_matrix = {}

    # Simple counter, starting at zero
    count = 0

    # For every file in the directory 
    for file in tqdm(directory):
        
        # Construct the GMM input 
        gmm_input = list()

        # Construct the output matrix dictionary 
        output_matrix_dic = {}

        # Load the pickle file 
        bayes_matrix[count] = pickle.load(open(file, 'rb'))
        
        # Set the file name 
        bayes_matrix[count]['filename'] = file
        
        # 
        Bayes1 = bayes_matrix[count]

        # For each parameter that will be deconvolved 
        for parameter in parameters_to_deconvolve:
            
            # Get the HiD parameter from the dictionary 
            hid_parameter = Bayes1[parameter]            
            
            # Set all the nan values to Zero 
            hid_parameter[np.where(np.isnan(hid_parameter))] = 0
            
            # Find the index where the hid_parameter is significant 
            index = np.where(hid_parameter > 1e-10)
            
            # use to reduce the data to fit
            A = np.squeeze(np.asarray(hid_parameter[index]))
            A = np.random.choice(A, 2000)                                 
            
            gmm_input = A.reshape(-1, 1)

            output_variable, output_matrix = applyGMM_functions.applyGMMfun(gmm_input, number_distributions)

            output_matrix_dic[parameter] = output_matrix
        
        # Update the deconvolution matrix 
        bayes_matrix[count]['Deconvolution'] = output_matrix_dic
        
        # Next file 
        count = count + 1 

    # Return tha analysis matrix 
    return bayes_matrix

def applyGMMconstrained_dir(listdir,parameters2decon,DistributionType,numDist):
    

    BayesMat={}
    count=0
    for filename in tqdm(listdir):
        GMM_input=[]
        outputmat={}

        BayesMat[count]=pickle.load(open(filename,'rb'))
        BayesMat[count]['filename']=filename
        Bayes1=BayesMat[count]

        count2=0
    
        for parameter2analyse in parameters2decon:
            HiD_parameter=Bayes1[parameter2analyse]            
            HiD_parameter[np.where(np.isnan(HiD_parameter))]=0
            index=np.where(HiD_parameter>1e-10)
            A = np.squeeze(np.asarray(HiD_parameter[index]))
            GMM_input = A.reshape(-1, 1)          
            outmat=applyGMMconstrained_fitout_functions.applyGMMfun(GMM_input,DistributionType[count2],numDist[count2])
            outputmat[parameter2analyse]=outmat
            outputmat[parameter2analyse]['GMM_input']=GMM_input
            count2=count2+1

        BayesMat[count]['Deconvolution']=outputmat
            
        count=count+1 
    
    return BayesMat

def generateplots_TestGMM(pathBayesCells_Plots,BayesMat, parameters,nbins,showplots):

    import numpy as np
    import pickle
    from tqdm import tqdm
    import os
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.stats import iqr
    from matplotlib import colors

    import seaborn
    from matplotlib import pyplot
    import matplotlib.pyplot as pyplot
    import matplotlib.font_manager as font_manager
    from matplotlib.ticker import FuncFormatter

    verify_plotting_packages()
    
    font_size = 14
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'True'
    pyplot.rcParams['grid.linewidth'] = 1.0
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'Helvetica LT Std'
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 1.0
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '10'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '1'

    for i in tqdm(range(len(BayesMat))):

        # Get the file name 
        file_name = BayesMat[i]['filename']
        
        # File name without extension 
        filename_without_ext = os.path.splitext(file_name)[0]

        colors = ['r', 'g', 'b']
        figure_width = 8
        figure_height = 3
        # Create the new plot 
        fig, axs = plt.subplots(1, len(parameters), figsize=(figure_width, figure_height))
        fig.clf
        fig.suptitle('Result') #'Filename: '+os.path.basename(filename_without_ext), fontsize=10)
 
        # For every parameter that should be deconvolved 
        for i_paramter in range(len(parameters)):

            # Get the parameter that needs analysis 
            parameter2analyse = parameters[i_paramter]
            
            # Lists 
            xdata = list()
            x = list()

            # MAGIC
            xdata = BayesMat[i][parameter2analyse].reshape(-1, 1)
            xdata[np.where(np.isnan(xdata))] = 0
            xdata = xdata[np.where(xdata > 1e-5)]

            # Construct the histogram 
            n, bins, patches = axs[i_paramter].hist(xdata, edgecolor=colors[i_paramter], color=colors[i_paramter], density=True, bins=nbins, alpha=0.3)
            x = np.arange(min(bins), max(bins), bins[1] - bins[0])

            
            weights = BayesMat[i]['Deconvolution'][parameter2analyse]['weights']
            mu = BayesMat[i]['Deconvolution'][parameter2analyse]['mu']
            sigma = BayesMat[i]['Deconvolution'][parameter2analyse]['sigma']
            DistributionType = BayesMat[i]['Deconvolution'][parameter2analyse]['DistributionType']
            number_populations = BayesMat[i]['Deconvolution'][parameter2analyse]['number_populations']
            model0 = BayesMat[i]['Deconvolution'][parameter2analyse]['model']
            
            axs[i_paramter].set_title('') #'DistType: '+DistributionType + ', # Populations: '+str(number_populations),fontsize=6)
            axs[i_paramter].set_xlabel(parameter2analyse, fontsize=10)
            
            axs[i_paramter].xaxis.set_tick_params(labelsize=font_size)
            axs[i_paramter].yaxis.set_tick_params(labelsize=font_size)

            axs[i_paramter].xaxis.set_visible(True)
            axs[i_paramter].yaxis.set_visible(True)

            # Adjust the spines
            axs[i_paramter].spines["bottom"].set_color('black')
            axs[i_paramter].spines['bottom'].set_linewidth(1)
            axs[i_paramter].spines["left"].set_color('black')
            axs[i_paramter].spines['left'].set_linewidth(1)

            # Plot the ticks
            axs[i_paramter].tick_params(axis='x', width=1, which='both', bottom=True)
            axs[i_paramter].tick_params(axis='y', width=1, which='both', left=True)

            xticks = sample_range(min(bins), max(bins), 3)
            axs[i_paramter].set_xlim(xticks[0], xticks[-1])
            axs[i_paramter].set_xticks(xticks)

            import math
            difference = axs[i_paramter].get_ylim()[1] - axs[i_paramter].get_ylim()[0]
            yticks = sample_range(axs[i_paramter].get_ylim()[0], axs[i_paramter].get_ylim()[1] + (difference * 0.1), 4)
            axs[i_paramter].set_ylim(yticks[0], yticks[-1])
            axs[i_paramter].set_yticks(yticks)

            #axs[i_paramter].set_ylim(yticks[0], yticks[-1])

            
            tempval=np.zeros(x.shape)

            if DistributionType == 'normal':
                for d in range(int(number_populations)):
                    axs[i_paramter].plot(x, weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d]))
                    tempval= tempval+weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d])
                axs[i_paramter].plot(x, tempval,'k')

            else:
                if number_populations==1:
                    param = model0.parameters
                    axs[i_paramter].plot(x, weights[0]*applyGMM_functions.lognormal(x,param[0],param[1]))
                else:
                    for d in range(int(number_populations)):           
                        param = model0.distributions[d].parameters
                        axs[i_paramter].plot(x, weights[d]*applyGMM_functions.lognormal((x), param[0],param[1]))
                        tempval= tempval+weights[d]*applyGMM_functions.lognormal((x), param[0],param[1])
                    axs[i_paramter].plot(x, tempval,'k')

            fig.tight_layout(pad=0.5)
        
        fig.savefig(pathBayesCells_Plots+os.path.basename(filename_without_ext)+'.png', dpi=300, bbox_inches='tight', pad_inches=0)
        if not(showplots):
            close(fig)
    

def generatetable_TestGMM(pathBayesCells, BayesMat, parameters2decon):
    
    row_labels=['normal','log-normal']
    df2={}
    tot=len(BayesMat)
    
    PopMat=np.zeros(len(parameters2decon)*tot)

    counter=0

    for j in range(len(parameters2decon)):
        for i in range(tot):
            PopMat[counter]=BayesMat[i]['Deconvolution'][parameters2decon[j]]['number_populations']    
            counter=counter+1
    
    maxPop=int(max(PopMat))
    
    column_labels = [str(x) for x in range(1,maxPop+1,1)]

    fig, ax = plt.subplots(len(parameters2decon))
    fig.clf
    fig.suptitle('Results_GMM_Multiple, Number of cells: '+str(tot), fontsize=10)

    for j in range(len(parameters2decon)):
        table=np.zeros((2,maxPop))

        dfTitle=pd.DataFrame([parameters2decon[j]])
        if j==0:
            dfTitle.to_csv(pathBayesCells+'Results_GMM_Multiple.csv', index=False, header=False)
        else:
            dfTitle.to_csv(pathBayesCells+'Results_GMM_Multiple.csv', mode='a',index=False, header=False)

        for i in range(tot):
            DistributionType=BayesMat[i]['Deconvolution'][parameters2decon[j]]['DistributionType']
            DistributionType=BayesMat[i]['Deconvolution'][parameters2decon[j]]['DistributionType']
            
            if DistributionType == 'normal':
                row=0
            else: 
                row=1
            
            col=BayesMat[i]['Deconvolution'][parameters2decon[j]]['number_populations']-1

            table[row,col]=table[row,col]+1

        table=table/tot 
        table=np.around(table,decimals=3)

        df= pd.DataFrame(table,index=row_labels,columns=column_labels)
        df2[j]=pd.concat([pd.concat([df],keys=['number_populations'], axis=1)], keys=['Dist_Type'])
        df2[j].to_csv(pathBayesCells+'Results_GMM_Multiple.csv', mode='a')        

        ax[j].table(cellText = df2[j].values,rowLabels = df2[j].index,colLabels = df2[j].columns,loc = "center")
        ax[j].set_title(parameters2decon[j])
        ax[j].axis("off")


def generateplots_GMMconstrained_fitout(pathBayesCells_Plots,BayesMat,parameters2decon,nbins,Sel_DistributionType,Sel_numDist,showplots):

    print('Verify Package!')

    verify_plotting_packages()

    font_size = 10
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linestyle'] = '--'
    pyplot.rcParams['grid.linewidth'] = 1.0
    pyplot.rcParams['grid.color'] = 'gray'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'Helvetica LT Std'
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 1.0
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '10'
    pyplot.rcParams['ytick.major.pad'] = '0'
    pyplot.rcParams['axes.edgecolor'] = '1'

   
    
    for i in tqdm(range(len(BayesMat))):

        filename=BayesMat[i]['filename']
        filename_without_ext = os.path.splitext(filename)[0]
        
        colors = ['r', 'g', 'b']
        figure_width = 8
        figure_height = 3

        fig, axs = plt.subplots(1, len(parameters2decon), figsize=(figure_width, figure_height))
        fig.clf
        fig.suptitle('Results_constrained. Filename: '+os.path.basename(filename_without_ext), fontsize=10)

        
        for count3 in range(len(parameters2decon)):
            parameter2analyse=parameters2decon[count3]
            xdata=[]
            x=[]

            xdata=BayesMat[i][parameter2analyse].reshape(-1, 1)
            xdata[np.where(np.isnan(xdata))]=0
            xdata=xdata[np.where(xdata>1e-10)]

            n,bins,patches=axs[count3].hist(xdata, edgecolor=colors[i], color=colors[i], density=True, bins=nbins, alpha=0.3);
            x=arange(min(bins),max(bins),bins[1]-bins[0])

            weights= BayesMat[i]['Deconvolution'][parameter2analyse]['weights']
            mu=BayesMat[i]['Deconvolution'][parameter2analyse]['mu']
            sigma=BayesMat[i]['Deconvolution'][parameter2analyse]['sigma']
            DistributionType=Sel_DistributionType[count3]
            number_populations=Sel_numDist[count3]
            model0=BayesMat[i]['Deconvolution'][parameter2analyse]['model']
            
            axs[count3].set_title('Dist Type: '+DistributionType + ', # Populations: '+str(number_populations),fontsize=6)
            #axs[count3].set_xlabel(parameter2analyse, fontsize=8)
            
            #axs[count3].xaxis.set_tick_params(labelsize=10)
            #axs[count3].yaxis.set_tick_params(labelsize=10)

            #

            #axs[count3].set_title('') #'DistType: '+DistributionType + ', # Populations: '+str(number_populations),fontsize=6)
            axs[count3].set_xlabel(parameter2analyse, fontsize=10)
            
            axs[count3].xaxis.set_tick_params(labelsize=font_size)
            axs[count3].yaxis.set_tick_params(labelsize=font_size)

            axs[count3].xaxis.set_visible(True)
            axs[count3].yaxis.set_visible(True)

            # Adjust the spines
            axs[count3].spines["bottom"].set_color('black')
            axs[count3].spines['bottom'].set_linewidth(1)
            axs[count3].spines["left"].set_color('black')
            axs[count3].spines['left'].set_linewidth(1)

            # Plot the ticks
            axs[count3].tick_params(axis='x', width=1, which='both', bottom=True)
            axs[count3].tick_params(axis='y', width=1, which='both', left=True)

            xticks = sample_range(min(bins), max(bins), 3)
            axs[count3].set_xlim(xticks[0], xticks[-1])
            axs[count3].set_xticks(xticks)

            import math
            difference = axs[count3].get_ylim()[1] - axs[count3].get_ylim()[0]
            yticks = sample_range(axs[count3].get_ylim()[0], axs[count3].get_ylim()[1] + (difference * 0.1), 4)
            axs[count3].set_ylim(yticks[0], yticks[-1])
            axs[count3].set_yticks(yticks)
            #
            
            tempval=np.zeros(x.shape)

            if DistributionType == 'normal':
                for d in range(int(number_populations)):
                    axs[count3].plot(x, weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d]))
                    tempval= tempval+weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d])
                axs[count3].plot(x, tempval,'k')

            else:
                if number_populations==1:
                    param = model0.parameters
                    axs[count3].plot(x, weights[0]*applyGMM_functions.lognormal(x,param[0],param[1]))
                else:
                    for d in range(int(number_populations)):           
                        param = model0.distributions[d].parameters
                        axs[count3].plot(x, weights[d]*applyGMM_functions.lognormal((x), param[0],param[1]))
                        tempval= tempval+weights[d]*applyGMM_functions.lognormal((x), param[0],param[1])
                    axs[count3].plot(x, tempval,'k')

            fig.tight_layout(pad=0.5)
        
        fig.savefig(pathBayesCells_Plots+os.path.basename(filename_without_ext)+'.png')
        if not(showplots):
            close(fig)
        
    return


def generate_plots_stats_decon(BayesMatSel,param,pathBayesCells_Populations_Plots,showplots):

    print('Verify Package!')

    verify_plotting_packages()
    font_size = 10
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linewidth'] = 1.0
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'Helvetica LT Std'
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 1.0
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '10'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '1'
    
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]

    labels = BayesMatSel['Deconvolution'][param]['labels']
    unique_labels = np.unique(labels)
    thresh = []

    for label in unique_labels:
        data_in_label = BayesMatSel['Deconvolution'][param]['GMM_input'][labels==label]
        thresh.append( np.max(data_in_label))

    thresh = np.sort(thresh)[:-1] # the last entry is not actually a threshold

    # map distributions back to nucleus
    labels_map = np.zeros(BayesMatSel[param].shape, dtype=int)
    numPop = len(thresh)+1
    for t in range(numPop):
        assigned_label = t+1    
        if t == 0:
            labels_map[BayesMatSel[param]<=thresh[t]] = assigned_label
        elif t > 0 and t<numPop-1:
            labels_map[ np.logical_and(BayesMatSel[param]>thresh[t-1], BayesMatSel[param]<=thresh[t]) ] = assigned_label
        else:
            labels_map[BayesMatSel[param]>thresh[t-1]] = assigned_label
    labels_map[BayesMatSel[param]==0] = 0
    
    #fig,ax=plt.subplots(1,2,figsize=[10,5])
    fig,ax=plt.subplots(1,2)
        
    listcolors=['w','g','b','purple','r','greenyellow']
    cmap = colors.ListedColormap(listcolors[0:numPop+1])

    img1=ax[0].imshow(labels_map, interpolation='nearest',cmap=cmap,origin='lower')


    ax[0].xaxis.set_visible(True)
    ax[0].yaxis.set_visible(True)

    # Adjust the spines
    ax[0].spines["bottom"].set_color('black')
    ax[0].spines['bottom'].set_linewidth(1)
    ax[0].spines["left"].set_color('black')
    ax[0].spines['left'].set_linewidth(1)

    # Plot the ticks
    ax[0].tick_params(axis='x', width=1, which='both', bottom=True)
    ax[0].tick_params(axis='y', width=1, which='both', left=True)








    cbar=fig.colorbar(img1,ax=ax[0],spacing='proportional',orientation='horizontal',boundaries=[-0.5] + bounds[0:numPop+1] + [numPop+0.5])
    labels_cbar = np.arange(0, numPop+1, 1)
    loc = labels_cbar
    cbar.set_ticks(loc)
    ax[0].set_title(param)
    
    # compute stats
    stats = {
        'means': [],
        'medians': [],
        'stds': [],
        'iqrs': [],
        }

    table0=np.zeros((4,numPop))

    for t in range(numPop):
        assigned_label = t+1
        data_in_label = BayesMatSel[param][labels_map==assigned_label]
        stats['means'].append( np.nanmean(data_in_label) )
        stats['medians'].append( np.nanmedian(data_in_label) )
        stats['stds'].append( np.nanstd(data_in_label) )
        stats['iqrs'].append( iqr(data_in_label) )
        table0[0,t]=np.nanmean(data_in_label)
        table0[1,t]=np.nanmedian(data_in_label)
        table0[2,t]=np.nanstd(data_in_label)
        table0[3,t]=iqr(data_in_label)

    #print(stats)

    row_labels=['mean','median','std','iqr']
    #column_labels=['1','2']
    column_labels = [str(x) for x in range(1,numPop+1,1)]
    df= pd.DataFrame(table0,index=row_labels,columns=column_labels)
    #column_labels=['1','2','3']
    #print(df)

    rounded_df = df.round(decimals=8)

    ax[1].table(cellText = rounded_df.values,rowLabels = rounded_df.index,colLabels = rounded_df.columns,loc='center')
    #ax[1].set_title(param)
    ax[1].axis('off')

    filename=BayesMatSel['filename']
    filename_without_ext = os.path.splitext(filename)[0]

    fig.suptitle('Statistics Populations after Deconvolution. Filename: '+os.path.basename(filename_without_ext), fontsize=10)

    pathBayesCells_Populations_Plots

    fig.tight_layout(pad=0.5)
        
    fig.savefig(pathBayesCells_Populations_Plots+os.path.basename(filename_without_ext)+'_Populations_'+param+'.png')
    if not(showplots):
        close(fig)

    filename_csv=pathBayesCells_Populations_Plots+os.path.basename(filename_without_ext)+'_Populations_'+param+'.csv'

    dfTitle=pd.DataFrame([param])
    dfTitle.to_csv(filename_csv, index=False, header=False)
    df.to_csv(filename_csv, mode='a')
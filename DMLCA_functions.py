# -*- coding: utf-8 -*-
"""
Created in 2017 in Python 3.5.2

@author: Tom Lauriks
Structure of this document follows an example of Michiel Stock.

Functions for the notebook dynamicModellingPlusLCA_notebook.ipynb
"""

# LOAD PACKAGES
# -------------

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

# FUNCTIONS
# -------------

def get_subset(dataset,columns):
    '''
    Selects columns in a pandas dataframe, as specified by the strings in columns. Returns a new dataframe with only these columns.
    Inputs:
        - dataset: Pandas dataframe where columns need to be selected. 
        - columns: columns from the dataset that need to be selected. List with a variable amount of strings. The function will 
        check if the column names in dataset start with these strings and add them to the output dataframe if so. The columns in 
        the output seem to be ordered exactly as in the input dataset, independently of the ordering of the strings given as 
        arguments.        
    Outputs:
        - subset: Pandas dataframe containing only the selected columns.
    '''
    columns=[col for col in list(dataset) for column in columns if col.startswith(column)]
    # Shorter way to do this, but the startswith interferes with the implementation:
    #[y for y in a if y in b]
    subset=dataset[columns].copy()
    return subset

def integrate(data,time_specifications):
    '''
    When this function is given a pandas dataframe with 1 column = 'time', it will integrate all other columns 
    over the time column. Most other columns in the data written out by WEST contain mass flow rates. Let F 
    denote mass flow rate, then integration occurs according to the formula:
    \int^{t}_{t=0} Fdt \approx \sum^{N}_{i=1} (F_{t_i}+F{t_{i-1}})/2 \cdot \Delta t_i.
    It can be chosen whether integration over the whole time span is carried out, or integration is carried out
    over smaller timespans that are a multiple of the time unit that WEST uses to write out subsequent data points.
    Inputs:
        - data: pandas dataframe with variables that need to be integrated. 
        - time_specifications: List containing two items:
            -> First element: True or False. False: integration over entire time span. True: entire time span of 
            data is divided in smaller time intervals. In each of these subsequent time intervals, integration
            will occur.
            -> Second element: If the first element in time_specifications is True, in the second element an integer
            needs to be given as argument. This integer determines the time intervals in which the entire time
            span will be divided. E.g.: the integer is 5. Then, integration will occur over every 5 time points
            written out by WEST.
    Outputs:
        - dataset_int: Integrated data set, where the integrated values are positioned next to the value 
        of the time at the end of the timespan over which they were integrated.
        - indices: Indices marking the positions - in the original dataset - of the time points at the end of
        each timespan over which integration occurred. E.g., if integration occurred over the whole 
        time span, indices will contain only the last index of the original pandas dataframe.
    '''
    ###Assigning the arguments to workable items.
    #Copying because otherwise this will modify the original dataframe given as argument! (SettingWithCopy 
    #warning)
    dataset=data.copy()
    time_intervals=time_specifications[0]
    ###First all separate values for (F_{t_i}+F{t_{i-1}})/2 \cdot \Delta t_i will be calculated.
    dataset.loc[1:,dataset.columns!='time']=(\
    dataset.loc[1:,dataset.columns!='time'].values+\
    dataset.loc[:len(dataset)-2,dataset.columns!='time'].values)*(\
    dataset.loc[1:,'time'].values.reshape((-1,1))-\
    dataset.loc[:len(dataset)-2,'time'].values.reshape((-1,1)))/2
    ###Next, the summation\sum^{N}_{i=1} (F_{t_i}+F{t_{i-1}})/2 \cdot \Delta t_i will be made. 
    if not time_intervals:#Over the whole time span.
        #Preallocate dataframe to store result. Only 2 rows needed: 1 at t=0 and 1 at t=t_end.
        dataset_int = pd.DataFrame(np.zeros((2,np.shape(dataset)[1])),\
                                   columns=dataset.keys())
        #Store begin and end time.
        dataset_int.loc[0,'time']=dataset.loc[0,'time']
        dataset_int.loc[1,'time']=dataset['time'].iloc[-1]
        #Summation over entire time span for each variable (each column is a variable) of the above calculated
        #values of F_{t_i}+F{t_{i-1}})/2 \cdot \Delta t_i
        dataset_int.loc[1,dataset_int.columns!='time']=dataset.loc[1:,dataset_int.columns!='time'].sum(axis=0)
        indices=[len(dataset)-1]
        return dataset_int, indices
    else:#Over the specified time intervals.
        time_units=time_specifications[1]
        #If the number of time points in dataset is a multiple of time_units:
        if (len(dataset)-1)%time_units==0:
            #Make a list containing the indices of the time points in the original data that mark the
            #time intervals in which integration will occur.
            indices=list(np.arange(time_units,len(dataset)-1+time_units,time_units))
            #Preallocate dataframe to store result.
            dataset_int = pd.DataFrame(np.zeros((len(indices)+1,np.shape(dataset)[1])),\
                                   columns=dataset.keys())
            dataset_int.loc[1:,'time']=dataset.loc[indices,'time'].values
            dataset_int.loc[0,'time']=dataset.loc[0,'time']
            #Make the summation of the above calculated values of F_{t_i}+F{t_{i-1}})/2 \cdot \Delta t_i 
            #for each time interval (and for each variable)
            intermediate=np.sum(dataset.loc[1:,dataset.columns!='time'].values.T.\
                                       reshape((-1,time_units)),axis=1)
            #Transform the calculated result to the right shape and store in resulting dataframe
            dataset_int.loc[1:,dataset_int.columns!='time']=intermediate.reshape((1,-1)).\
            reshape((np.shape(dataset)[1]-1,len(indices))).T
        else:#If not a multiple an additional time interval is present at the end of the data.
            indices=list(np.arange(time_units,len(dataset)-1,time_units))
            dataset_int = pd.DataFrame(np.zeros((len(indices)+2,np.shape(dataset)[1])),\
                                   columns=dataset.keys())
            intermediate=np.sum(dataset.loc[1:indices[-1],dataset.columns!='time'].values.T.\
                                       reshape((-1,time_units)),axis=1)
            dataset_int.loc[1:len(dataset_int)-2,dataset_int.columns!='time']=intermediate.reshape((1,-1)).\
            reshape((np.shape(dataset)[1]-1,len(indices))).T
            #Add last integrated time interval (that is not a multiple of time_units) and index
            dataset_int.loc[len(dataset_int)-1,dataset_int.columns!='time']=np.sum\
            (dataset.loc[indices[-1]+1:,dataset.columns!='time'].values,axis=0)
            indices.append(len(dataset)-1)
            dataset_int.loc[1:,'time']=dataset.loc[indices,'time'].values
            dataset_int.loc[0,'time']=dataset.loc[0,'time']
        return dataset_int,indices


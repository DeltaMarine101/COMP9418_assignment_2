#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:34:43 2020

@author: pablopacheco
"""

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries
import numpy as np
import pandas as pd
import scipy as sp
import heapq as pq
import matplotlib as mp
import math
from itertools import product, combinations
from collections import OrderedDict as odict
from graphviz import Digraph
from tabulate import tabulate



# Dependencies for transition probabilities
previous_G = {
    'r1' : ['r1','r2','r3'],
    'r2' : ['r1', 'r2','r4'],
    'r3' : ['r1', 'r3','r7'],
    'r4' : ['r2', 'r4','r8'],
    'r5' : ['r5','r6', 'r9', 'c3'],
    'r6' : ['r5','r6','c3'],
    'r7' : ['r3', 'r7','c1'],
    'r8' : ['r4', 'r8','r9'],
    'r9' : ['r5', 'r8','r9', 'r13'],
    'r10': ['r10','c3'],
    'r11': ['r11','c3'],
    'r12': ['r12','r22', 'outside'],
    'r13': ['r9', 'r13','r24'],
    'r14': ['r14','r24'],
    'r15': ['r15','c3'],
    'r16': ['r16','c3'],
    'r17': ['r17','c3'],
    'r18': ['r18','c3'],
    'r19': ['r19','c3'],
    'r20': ['r20','c3'],
    'r21': ['r21','c3'],
    'r22': ['r12', 'r22','r25'],
    'r23': ['r23','r24'],
    'r24': ['r13', 'r14', 'r23','r24'],
    'r25': ['r22', 'r25','r26'],
    'r26': ['r25', 'r26','r27'],
    'r27': ['r26', 'r27','r32'],
    'r28': ['r28','c4'],
    'r29': ['r29','r30', 'c4'],
    'r30': ['r29','r30'],
    'r31': ['r31','r32'],
    'r32': ['r27', 'r31','r32', 'r33'],
    'r33': ['r32','r33'],
    'r34': ['r34','c2'],
    'r35': ['r35','c4'],
    'c1' : ['r7', 'r25','c1', 'c2'],
    'c2' : ['r34', 'c1','c2', 'c4'],
    'c3' : ['r5', 'r6', 'r10', 'r11', 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'c3','o1'],
    'c4' : ['r28', 'r29', 'r35', 'c2','c4', 'o1'],
    'o1' : ['c3', 'c4','o1'],
    'outside': ['r12','outside']
}

#simplify graph
for k in previous_G.keys():
    previous_G[k]=[k]


# Sensor locations
urel_sens_loc = {
    'unreliable_sensor1': 'o1',
    'unreliable_sensor2': 'c3',
    'unreliable_sensor3': 'r1',
    'unreliable_sensor4': 'r24'
}

rel_sens_loc = {
    'reliable_sensor1': 'r16',
    'reliable_sensor2': 'r5',
    'reliable_sensor3': 'r25',
    'reliable_sensor4': 'r31'
}

door_sens_loc = {
    'door_sensor1': ['r8', 'r9'],
    'door_sensor2': ['c1', 'c2'],
    'door_sensor3': ['r26', 'r27'],
    'door_sensor4': ['r35', 'c4']
}

all_sensors={
    'unreliable_sensor1': ['o1'],
    'unreliable_sensor2': ['c3'],
    'unreliable_sensor3': ['r1'],
    'unreliable_sensor4': ['r24'],
    'reliable_sensor1': ['r16'],
    'reliable_sensor2': ['r5'],
    'reliable_sensor3': ['r25'],
    'reliable_sensor4': ['r31'],
    'door_sensor1': ['r8', 'r9'],
    'door_sensor2': ['c1', 'c2'],
    'door_sensor3': ['r26', 'r27'],
    'door_sensor4': ['r35', 'c4']
}

rob_list = ['robot1', 'robot2']

# Creating reverse lookup dict
sens_rooms = {}
for sens, locs in all_sensors.items():
    for i in locs: sens_rooms[i] = sens

#Creating non_sens_rooms
list_non_sens_rooms=[]
for k in previous_G.keys():
    if k not in sens_rooms.keys():
        list_non_sens_rooms.append(k)

#Function to learn the outcomespace
def learn_outcome_space(data):
    outcomeSpace=dict()
    for i in data.keys():
        outcomeSpace[i] = tuple(np.unique(data[i]))
        # previous timestep nodes
        outcomeSpace[i + '_t-1'] = outcomeSpace[i]
    return outcomeSpace

#Load data
data = pd.read_csv('data.csv')
data_numpy = data.to_numpy()
data_cols = list(data.columns)

all_rooms = list(previous_G.keys())
#We are going to consider just if a room is empty or not (we don't care about the exact number of people)
data_processed = {}
for i in all_rooms + list(door_sens_loc.keys()):
    data_processed[i] = (data_numpy[:,data_cols.index(i)] > 0)

for i in list(urel_sens_loc.keys()) + list(rel_sens_loc.keys()):
    data_processed[i] = (data_numpy[:,data_cols.index(i)] == "motion")

#In the case of a space: the value True means there is at least one person and False there is no person at all
#In the case of a sensor: the value True means there was detected movement and False it wasn't
outcomeSpace = learn_outcome_space(data_processed)

#function from tutorial to print a factor
def printFactor(f):
    """
    argument
    `f`, a factor to print on screen
    """
    # Create a empty list that we will fill in with the probability table entries
    table = list()

    # Iterate over all keys and probability values in the table
    for key, item in f['table'].items():
        # Convert the tuple to a list to be able to manipulate it
        k = list(key)
        # Append the probability value to the list with key values
        k.append(item)
        # Append an entire row to the table
        table.append(k)
    # dom is used as table header. We need it converted to list
    dom = list(f['dom'])
    # Append a 'Pr' to indicate the probabity column
    dom.append('Pr')
    print(tabulate(table,headers=dom,tablefmt='orgtbl'))

#Function from tutorial to hep to construct the table probablities
def allEqualThisIndex(dict_of_arrays, **fixed_vars):
    """
    Helper function to create a boolean index vector into a tabular data structure,
    such that we return True only for rows of the table where, e.g.
    column_a=fixed_vars['column_a'] and column_b=fixed_vars['column_b'].

    This is a simple task, but it's not *quite* obvious
    for various obscure technical reasons.

    It is perhaps best explained by an example.

    >>> all_equal_this_index(
    ...    {'X': [1, 1, 0], Y: [1, 0, 1]},
    ...    X=1,
    ...    Y=1
    ... )
    [True, False, False]
    """
    # base index is a boolean vector, everywhere true
    first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
    index = np.ones_like(first_array, dtype=np.bool_)
    for var_name, var_val in fixed_vars.items():
        index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
    return index

#Function to create the transition and emission probability tables
def estProbs(data, var_name, parent_names, outcomeSpace, parent_offest=0):
    """
    Calculate a dictionary probability table by ML given
    `data`, a dictionary or dataframe of observations
    `var_name`, the column of the data to be used for the conditioned variable and
    `parent_names`, a tuple of columns to be used for the parents and
    `outcomeSpace`, a dict that maps variable names to a tuple of possible outcomes
    Return a dictionary containing an estimated conditional probability table.
    """
    var_outcomes = outcomeSpace[var_name]
    parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
    # cartesian product to generate a table of all possible outcomes
    all_parent_combinations = product(*parent_outcomes)

    # Smoothing
    alpha = 1
    prob_table = odict()

    # Changed to only output the probability that there are people in the room p, and so P(0) = 1 - p
    # This makes tables much smaller and keeps the exact same information since outcome space is binary
    for i, parent_combination in enumerate(all_parent_combinations):
        parent_vars = dict(zip(parent_names, parent_combination))
        #print(parent_vars)
        parent_index = allEqualThisIndex(data, **parent_vars)
        ########we care for the previous state only, so we delete the last row#########
        parent_index=(parent_index, parent_index[:-parent_offest])[parent_offest > 0]
        #print('parent_index',len(parent_index),parent_index)
        #print('var_outcome:',var_outcome)
        var_index = data[var_name][parent_offest:]
        ########we need to consider from the second state, so we delete the first row#########
        #print('var_index',len(var_index),var_index)

        p = ((var_index & parent_index).sum()+alpha)/(parent_index.sum() + alpha*len(var_outcomes))
        prob_table[tuple(list(parent_combination)+[1])] = p
        prob_table[tuple(list(parent_combination)+[0])] = 1 - p

    # 'r16t-1' Denotes previous time as opposed to current time 'r16'
    if parent_offest: parent_names = [i + '_t-1' for i in parent_names]

    return {'dom': tuple(list(parent_names)+[var_name]), 'table': prob_table}

#function from tutorial to calculate probability given an entry
def prob(factor, *entry):
    return factor['table'][entry]

def marginalize(f, var, outcomeSpace):
    """
    argument
    `f`, factor to be marginalized.
    `var`, variable to be summed out.
    `outcomeSpace`, dictionary with the domain of each variable

    Returns a new factor f' with dom(f') = dom(f) - {var}
    """
    # Let's make a copy of f domain and convert it to a list. We need a list to be able to modify its elements
    new_dom = list(f['dom'])

    # print(var,"!!!!", new_dom)
    new_dom.remove(var)            # Remove var from the list new_dom by calling the method remove().
    table = list()                 # Create an empty list for table. We will fill in table from scratch.
    for entries in product(*[outcomeSpace[node] for node in new_dom]):
        s = 0;                     # Initialize the summation variable s.

        # We need to iterate over all possible outcomes of the variable var
        for val in outcomeSpace[var]:
            # To modify the tuple entries, we will need to convert it to a list
            entriesList = list(entries)
            # We need to insert the value of var in the right position in entriesList
            entriesList.insert(f['dom'].index(var), val)

            p = prob(f, *tuple(entriesList))     # Calculate the probability of factor f for entriesList.
            s = s + p                            # Sum over all values of var by accumulating the sum in s.

        # Create a new table entry with the multiplication of p1 and p2
        table.append((entries, s))
    return {'dom': tuple(new_dom), 'table': odict(table)}

# Get all transition probabilities
tran_prob_table={}
for present, previous in previous_G.items():
    tran_prob_table[present] = estProbs(data_processed, present, previous, outcomeSpace, parent_offest=1)

# Get all emission probabilities
emis_prob_table={}
for sensor, location in all_sensors.items():
    emis_prob_table[sensor] = estProbs(data_processed, sensor, location, outcomeSpace)

#decompose door_sensors emission prob in order to get just one parent in the conditional prob
#now, the probability P(door_sensor1|r8) is going to be in emis_prob_table['door_sensor1_r8']
for k in door_sens_loc.keys():
    for i in door_sens_loc[k]:
        #create the empty dict
        emis_prob_table[str(k)+'_'+str(i)]={}
        emis_prob_table[str(k)+'_'+str(i)]['dom']=(str(i),str(k))
        emis_prob_table[str(k)+'_'+str(i)]['table']=odict()

        #marginalize over the parent that we want to extract
        extr_parent=set(emis_prob_table[k]['dom'])-set([i,k])
        marg_prob=marginalize(emis_prob_table[k],extr_parent.pop(),outcomeSpace)
        
        #normilize probabilities
        for j in range(2):
            prob_1=prob(marg_prob,j,0)
            prob_2=prob(marg_prob,j,1)
            s_probs=prob_1+prob_2
            emis_prob_table[str(k)+'_'+str(i)]['table'][(j,0)]=marg_prob['table'][(j,0)]/s_probs
            emis_prob_table[str(k)+'_'+str(i)]['table'][(j,1)]=marg_prob['table'][(j,1)]/s_probs

#Eliminate the old door_sensor tables
for k in door_sens_loc.keys():
    emis_prob_table.pop(k)

#Make a dictionary of emission probabilities with the state variable as a key
emis_prob_table_varstate={}
for k in emis_prob_table.keys():
    emis_prob_table_varstate[emis_prob_table[k]['dom'][0]]=emis_prob_table[k]

#Get all the initial probabilities which are going to be 1 for empty and 0 for not_empty
initial_prob_tables={}
for k in previous_G.keys():
    initial_prob_tables[k]={'dom':(k,),'table':odict([((False,),1.0) ,((True,),0.0),]) }
initial_prob_tables['outside']={'dom':(k,),'table':odict([((False,),0.0) ,((True,),1.0),]) }

#Function from tutorial to get a new outcomespace given some evidence
def evidence(var, e, outcomeSpace):
    """
    argument
    `var`, a valid variable identifier.
    `e`, the observed value for var.
    `outcomeSpace`, dictionary with the domain of each variable

    Returns dictionary with a copy of outcomeSpace with var = e
    """
    newOutcomeSpace = outcomeSpace.copy()
    newOutcomeSpace[var] = (e,)
    return newOutcomeSpace

#function from tutoral to calculate join probability given two factors
def join(f1, f2, outcomeSpace):
    """
    argument
    `f1`, first factor to be joined.
    `f2`, second factor to be joined.
    `outcomeSpace`, dictionary with the domain of each variable

    Returns a new factor with a join of f1 and f2
    """

    # First, we need to determine the domain of the new factor. It will be union of the domain in f1 and f2
    # But it is important to eliminate the repetitions
    common_vars = list(f1['dom']) + list(set(f2['dom']) - set(f1['dom']))

    # We will build a table from scratch, starting with an empty list.
    table = list()

    # The product iterator will generate all combinations of varible values
    # as specified in outcomeSpace. Therefore, it will naturally respect observed values
    # print("****", common_vars)
    # print(outcomeSpace)
    for entries in product(*[outcomeSpace[node] for node in common_vars]):

        # We need to map the entries to the domain of the factors f1 and f2
        entryDict = dict(zip(common_vars, entries))
        f1_entry = (entryDict[var] for var in f1['dom'])
        f2_entry = (entryDict[var] for var in f2['dom'])

        p1 = prob(f1, *f1_entry)
        p2 = prob(f2, *f2_entry)

        # Create a new table entry with the multiplication of p1 and p2
        table.append((entries, p1 * p2))
    return {'dom': tuple(common_vars), 'table': odict(table)}

#Function from tutorial to make queries, the only difference is that this function does not normalize at
#the end. Given that we are classifying, it is just needed to choose the most likely one. In addition, we are
#assuming that q_vars is going to be just one variable because we are doing classification.


def miniForwardOnline(f, transition, outcomeSpace):
    """
    argument 
    `f`, factor that represents the previous state of the chain.
    `transition`, transition probabilities from time t-1 to t.
    `outcomeSpace`, dictionary with the domain of each variable.
    
    Returns a new factor that represents the current state of the chain.
    """

    # Make a copy of f so we will not modify the original factor
    fPrevious = f.copy()                                          
    # Name of the random variable. f domain should be a list with a single element
    randVariable = fPrevious['dom'][0]                                     
    # Set the f_previous domain to be a list with a single variable name appended with '_t-1' to indicate previous time step
    #necesitas que dom de fprevious sea t-1 para que join lo asocie a t-1 en la transition table
    fPrevious['dom'] = (randVariable + '_t-1', )
    #print(fPrevious)
    # Make the join operation between fPrevious and the transition probability table
    fCurrent = join(fPrevious, transition, outcomeSpace)
    # Marginalize the randVariable_t-1
    fCurrent = marginalize(fCurrent, fPrevious['dom'][0], outcomeSpace)
    # Set the domain of fCurrent to be name of the random variable without time index
    fCurrent['dom'] = (randVariable, )
    return fCurrent


def miniForwardBatch(f, fTransition, outcomeSpace, n):
    """
    argument 
    `f`, factor that represents the previous state of the chain.
    `fTransition`, transition probabilities from time t-1 to t.
    `outcomeSpace`, dictionary with the domain of each variable.
    `n`, number of time updates
    
    Returns a new factor that represents the current state of the chain after n time steps.
    """

    # fCurrent is a copy of f, so we will not overwrite f in the for loop
    fCurrent = f.copy()
    for i in range(n):
        # Call miniForwardOnline to update fCurrent
        fCurrent = miniForwardOnline(fCurrent, fTransition, outcomeSpace)
        # Print fCurrent to debug the results
        printFactor(fCurrent)
        print()
    # return fCurrent
    return fCurrent

def convError(f1, f2, outcomeSpace):
    """
    argument 
    `f1`, factor with the current state probability distribution in the chain.
    `f2`, factor with the previous state probability distribution in the chain.
    `outcomeSpace`, dictionary with the domain of each variable.    
    
    Returns absolute error between f1 and f2.
    """
    return sum([abs(prob(f1, var) - prob(f2, var)) for var in outcomeSpace[f1['dom'][0]]])

def miniforwardConvergence(f, transition, outcomeSpace, n = 1000, eps = 0.00001):
    """
    argument 
    `f`, factor that represents the previous state of the chain.
    `transition`, transition probabilities from time t-1 to t.
    `outcomeSpace`, dictionary with the domain of each variable.
    `startDistribution`, initial state probability distribution.
    `n`, maximum number of time updates.
    `eps`, error threshold to determine convergence.
    
    Returns a new factor that represents the current state of the chain after n time steps or the convergence error is less than eps.
    """

    #print("  Iter Error\n------ --------")
    # fCurrent is a copy of f, so we will not overwrite f in the for loop
    fCurrent = f.copy()
    # Set an empty list of error, so we can plot the errors later
    errors = []
    for i in range(n):
        # Call miniForwardOnline to compute fNew
        fNew = miniForwardOnline(fCurrent, transition, outcomeSpace)
        # Calculate error between fNew and fCurrent using the convError function
        error = convError(fNew, fCurrent, outcomeSpace)
        # Halt the loop if the error is smaller than eps
        if (error < eps):
            break
        # Print a message every 10 iterations to inform the convergence progress
        #if (i % 10 == 0):
        #    print("%6d %1.6f" % (i, error))
        # Store the current error value in a list of error so we can plot it later
        errors.append(error)
        # Updates fCurrent as fNew
        fCurrent = fNew
    #Plot the errors
    #mp.pyplot.plot(errors, 'ro')
    #mp.pyplot.show()

    return fCurrent

#Get all the convergence probabilities
convergence_prob_tables=odict()
for k in previous_G.keys():
    convergence_prob_tables[k]=miniforwardConvergence(initial_prob_tables[k], tran_prob_table[k], outcomeSpace)
    #printFactor(convergence_prob_tables[k])


# print(convergence_prob_tables['r1']['table'][(1,)])
# print(convergence_prob_tables['r1']['table'][(0,)])

###############################################################################
##########
###############################################################################
#Function from tutorial to normalize
def normalize(f):
    """
    argument 
    `f`, factor to be normalized.
    
    Returns a new factor f' as a copy of f with entries that sum up to 1
    """ 
    table = list()
    sum = 0
    for k, p in f['table'].items():
        sum = sum + p
    for k, p in f['table'].items():
        table.append((k, p/sum))
    return {'dom': f['dom'], 'table': odict(table)}

#tutorial function to do forward with evidence
#actually, miniforwardonline could be mixed with this
def forwardOnline(f, transition, emission, stateVar, emissionVar, emissionEvi, outcomeSpace):
    """
    argument 
    `f`, factor that represents the previous state.
    `transition`, transition probabilities from time t-1 to t.
    `emission`, emission probabilities.
    `stateVar`, state (hidden) variable.
    `emissionVar`, emission variable.
    `emissionEvi`, emission observed evidence. If undef, we do only the time update
    `outcomeSpace`, dictionary with the domain of each variable.
    
    Returns a new factor that represents the current state.
    """

    # Set fCurrent as a copy of f
    fCurrent = f.copy();
    # Set the f_previous domain to be a list with a single variable name appended with '_t-1' to indicate previous time step
    fCurrent['dom'] = (stateVar + '_t-1', )       
    # Make the join operation between fCurrent and the transition probability table
    fCurrent = join(fCurrent, transition, outcomeSpace)
    # Marginalize the randVariable_t-1
    fCurrent = marginalize(fCurrent, fCurrent['dom'][0], outcomeSpace)
    # If emissionEvi == None, we will assume this time step has no observed evidence    
    if emissionEvi != None:
        # Set evidence in the form emissionVar = emissionEvi
        newOutcomeSpace = evidence(emissionVar, emissionEvi, outcomeSpace)
        # Make the join operation between fCurrent and the emission probability table. Use the newOutcomeSpace
        fCurrent = join(fCurrent, emission, newOutcomeSpace)
        # Marginalize emissionVar. Use the newOutcomeSpace
        fCurrent = marginalize(fCurrent, emissionVar, newOutcomeSpace) 
        # Normalize fCurrent, optional step
        fCurrent = normalize(fCurrent)
    # Set the domain of w to be name of the random variable without time index
    fCurrent['dom'] = (stateVar, )
    return fCurrent


#actions turn on or turn off the lights according to the stationary distributions
stationary_actions_dict={}
for i in range(1,36):
         stationary_actions_dict['lights'+str(i)]=('off','on')[convergence_prob_tables['r'+str(i)]['table'][(1,)] > convergence_prob_tables['r'+str(i)]['table'][(0,)]]

original_stationary_actions_dict= stationary_actions_dict.copy()
# Initial state - actually we can use just the sensored spaces because we are goingt o use the stationary distribution for the other rooms
state = initial_prob_tables.copy()


def get_action(sensor_data):
    global stationary_actions_dict
    global initial_prob_tables
    global tran_prob_table
    global emis_prob_table_varstate
    global outcomeSpace
    global state
    global sens_rooms
    global urel_sens_loc
    global rel_sens_loc
    global door_sens_loc
    global list_non_sens_rooms
    
    #go back to stationary probabilities (just fo non_sensored rooms)
    for i in list_non_sens_rooms:
        if i != 'outside':
            stationary_actions_dict['lights'+i.split('r')[1]]= original_stationary_actions_dict['lights'+i.split('r')[1]]
    
    
    #transform sensor_data
    for k in sensor_data.keys():
        if sensor_data[k]!=None:
            if k in list(urel_sens_loc.keys())+list(rel_sens_loc.keys()):
                sensor_data[k]=(0,1)[sensor_data[k]=='motion']
            elif k in door_sens_loc.keys():
                sensor_data[k]=(0,1)[sensor_data[k]>0]
                
    #we are not using all the rooms, so we could ignore the spaces which don't have lights        
    for i in state.keys():
        if i in sens_rooms.keys():
            state[i]=forwardOnline(state[i],tran_prob_table[i],emis_prob_table_varstate[i],i,sens_rooms[i],sensor_data[sens_rooms[i]],outcomeSpace)
            if i.startswith('r'):
                stationary_actions_dict['lights'+ i.split('r')[1]]=('off','on')[prob(state[i],0)<prob(state[i],1)]

                
    #use robots
    for i in ['robot1','robot2']:
        #info=sensor_data[i]
        if sensor_data[i] != None:
            seen_room=sensor_data[i].split(',')[0].partition("'")[2].partition("'")[0]
            num_pp=int(sensor_data[i].split(',')[1].strip().partition(')')[0])
            if seen_room.startswith('r'):
                if num_pp>0: 
                    stationary_actions_dict['lights'+seen_room.split('r')[1]]='on'
                    #we just want to modify the prob distribution for sensor rooms, we are using the stationary for the others
                    if seen_room not in list_non_sens_rooms:
                        state[seen_room]['table']=odict([((False,),0.0) ,((True,),1.0),])
                else:
                    stationary_actions_dict['lights'+seen_room.split('r')[1]]='off'
                    if seen_room not in list_non_sens_rooms:
                        state[seen_room]['table']=odict([((False,),1.0) ,((True,),0.0),])
                
    #some hard-code
    #at the beginning and at the end room12 should be on, bacause is next to outside
    
    
    
    
    return stationary_actions_dict










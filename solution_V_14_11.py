#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: Jeremie Kull    zID: z5208518

Name: Pablo Pacheco   zID: z5222810
'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries
import numpy as np
import pandas as pd
import scipy as sp
import heapq as pq
import matplotlib as mp
import math, time, random
from itertools import product, combinations
from collections import OrderedDict as odict
from graphviz import Digraph
from tabulate import tabulate


# Dependencies for transition probabilities (considering important interactions)
previous_G = {
    'r1' : ['r1', 'r2'],
    'r2' : ['r1', 'r2'],
    'r3' : ['r1', 'r3'],
    'r4' : ['r2', 'r4'],
    'r5' : ['r5', 'r6', 'r9'],
    'r6' : ['r6', 'c3'],
    'r7' : ['r3', 'r7'],
    'r8' : ['r8', 'r9', 'r5'],
    'r9' : ['r8', 'r9', 'r13'],
    'r10': ['r10', 'c3'],
    'r11': ['r11', 'c3', 'o1'],
    'r12': ['r12', 'r22', 'outside'],
    'r13': ['r9', 'r13', 'r8'],
    'r14': ['r14', 'r24'],
    'r15': ['r15', 'c3', 'o1'],
    'r16': ['r16', 'c3'],
    'r17': ['r17', 'c3'],
    'r18': ['r18', 'c3'],
    'r19': ['r19', 'c3', 'r5'],
    'r20': ['r20', 'c3'],
    'r21': ['r21', 'c3'],
    'r22': ['r22', 'r25', 'outside'],
    'r23': ['r23', 'r24'],
    'r24': ['r13', 'r23', 'r24'],
    'r25': ['r25', 'r26', 'outside'],
    'r26': ['r27', 'outside', 'r25'],
    'r27': ['r26', 'r27', 'r31'],
    'r28': ['r28', 'c4'],
    'r29': ['r29', 'r30', 'c4'],
    'r30': ['r29', 'r30'],
    'r31': ['r31', 'r32'],
    'r32': ['r31', 'r32', 'r33'],
    'r33': ['r32', 'r33', 'r27'],
    'r34': ['r34', 'c2', 'c1'],
    'r35': ['r35', 'c4'],
    'c1' : ['r7', 'c1', 'c2'],
    'c2' : ['c2', 'c4'],
    'c3' : ['c3', 'r20', 'c4'],
    'c4' : ['r28', 'r35', 'c2'],
    'o1' : ['c3', 'c4', 'o1'],
    'outside': ['r12', 'outside', 'r25']
}

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

all_sensors = {
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
for i in all_rooms:
    data_processed[i] = (data_numpy[:, data_cols.index(i)] > 0)

# Consider a door sensor 'active' only if 2 or more people have walked through it
for i in door_sens_loc.keys():
    data_processed[i] = (data_numpy[:, data_cols.index(i)] > 1)

for i in list(urel_sens_loc.keys()) + list(rel_sens_loc.keys()):
    data_processed[i] = (data_numpy[:, data_cols.index(i)] == "motion")

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

######################## GENERATE TABLES FROM DATA #############################
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
    # Greatly improves our result
    p = .95
    initial_prob_tables[k]={'dom':(k,),'table':odict([((False,),p) ,((True,),1-p),]) }
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



#This is a modification of the tutorial function. This is for markov chains which state variable depend on the
#the previous state of one or more state variables
def miniForwardOnline(f, transition, outcomeSpace):
    """
    argument
    'state_variable'(string), state variable whose factor is going to be calculated
    `f`, dictionary of factors (in the previous state of the chain) asociated with the state_variable
    `transition`, transition probabilities from time t-1 to t.
    `outcomeSpace`, dictionary with the domain of each variable.

    Returns a new factor that represents the current state of the chain.
    """

    # Make a copy of f so we will not modify the original factor
    fPrevious = f.copy()
    #Put t-1 to the previous domains
    for i in fPrevious.keys():
        fPrevious[i]['dom']=(i+'_t-1',)

    #Do a join of all the previous factors
    count=0
    for i in fPrevious.keys():
        if count==0:
            joint_prob=fPrevious[i]
            count +=1
        else:
            joint_prob=join(joint_prob,fPrevious[i],outcomeSpace)

    fCurrent=join(joint_prob,transition,outcomeSpace)

    #Then, we need to marginalize all the previous state variables
    for i in fPrevious.keys():
        fCurrent=marginalize(fCurrent,i+'_t-1',outcomeSpace)

    fCurrent=normalize(fCurrent)

    return fCurrent



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


def forwardOnlineEmission(f, transition, emission, stateVar, emissionVar, emissionEvi, outcomeSpace):
    """
    argument
    'state_variable'(string), state variable whose factor is going to be calculated
    `f`, dictionary of factors (in the previous state of the chain) asociated with the main state variable
    `transition`, transition probabilities from time t-1 to t of the main state variable
    `emission`, emission probabilities.
    `stateVar`, state (hidden) variable.
    `emissionVar`, emission variable.
    `emissionEvi`, emission observed evidence. If undef, we do only the time update
    `outcomeSpace`, dictionary with the domain of each variable.

    Returns a new factor that represents the current state of the chain.
    """
    # perform normal chain forward algorithm
    fCurrent = miniForwardOnline(f, transition, outcomeSpace)

    if emissionEvi != None:
        # Set evidence in the form emissionVar = emissionEvi
        newOutcomeSpace = evidence(emissionVar, emissionEvi, outcomeSpace)
        # Make the join operation between fCurrent and the emission probability table. Use the newOutcomeSpace
        fCurrent = join(fCurrent, emission, newOutcomeSpace)
        # printFactor(fCurrent)
        # Marginalize emissionVar. Use the newOutcomeSpace
        fCurrent = marginalize(fCurrent, emissionVar, newOutcomeSpace)
        # Normalize fCurrent, optional step
        fCurrent = normalize(fCurrent)

    return fCurrent



# Initial state
state = initial_prob_tables.copy()
previous_state=state.copy()
actions_dict = {'lights1': 'off', 'lights2': 'off', 'lights3': 'off', 'lights4': 'off', 'lights5': 'off', 'lights6': 'off', 'lights7': 'off', 'lights8': 'off', 'lights9': 'off', 'lights10': 'off', 'lights11': 'off', 'lights12': 'off', 'lights13': 'off', 'lights14': 'off', 'lights15': 'off', 'lights16': 'off', 'lights17': 'off', 'lights18': 'off', 'lights19': 'off', 'lights20': 'off', 'lights21': 'off', 'lights22': 'off', 'lights23': 'off', 'lights24': 'off', 'lights25': 'off', 'lights26': 'off', 'lights27': 'off', 'lights28': 'off', 'lights29': 'off', 'lights30': 'off', 'lights31': 'off', 'lights32': 'off', 'lights33': 'off', 'lights34': 'off', 'lights35':'off'}



d = .52 # default offset
offsets = [.50, d, .54, .48, d, .10, .45, -.10, .43, d, d,
    d, .45, d, .43, d, .65, .46, .25, .60, .40, .35, .65,
    .45, .60, -.15, .45, d, d, d, .65, .65, .43, .15, .6]

# Offset for prioritising lights off vs on for each room individually
offset = {}
for i in range(35):
    offset['r' + str(i + 1)] = offsets[i]

def get_action(sensor_data):
    global actions_dict
    global initial_prob_tables
    global tran_prob_table
    global emis_prob_table_varstate
    global outcomeSpace
    global state
    global previous_state
    global sens_rooms
    global urel_sens_loc
    global rel_sens_loc
    global door_sens_loc
    global list_non_sens_rooms
    global previous_G

    # Slightly offset probabilities based on electricity price
    elec = sensor_data['electricity_price']
    elec_weight = 0.55

    #transform sensor_data
    for k in sensor_data.keys():
        if sensor_data[k]!=None:
            if k in list(urel_sens_loc.keys())+list(rel_sens_loc.keys()):
                sensor_data[k]=(0,1)[sensor_data[k]=='motion']
            elif k in door_sens_loc.keys():
                sensor_data[k]=(0,1)[sensor_data[k]>0]

    for i in state.keys():
        # create dictionary with the factors of the state variables associated with the state variable i
        dict_variables={}
        for k in previous_G[i]:
            dict_variables[k]=previous_state[k]
        if i in sens_rooms.keys():
            state[i]=forwardOnlineEmission(dict_variables, tran_prob_table[i],emis_prob_table_varstate[i],i,sens_rooms[i],sensor_data[sens_rooms[i]],outcomeSpace)
        else:
            state[i]=miniForwardOnline(dict_variables, tran_prob_table[i], outcomeSpace)

        if i.startswith('r'):
            inde = prob(state[i], 0) <= prob(state[i], 1) + (1 - elec) * elec_weight + offset[i]
            actions_dict['lights' + i.split('r')[1]] = ('off', 'on')[inde]

    # use robots
    for i in ['robot1','robot2']:
        #info=sensor_data[i]
        if sensor_data[i] != None:
            seen_room=sensor_data[i].split(',')[0].partition("'")[2].partition("'")[0]
            num_pp=int(sensor_data[i].split(',')[1].strip().partition(')')[0])
            if seen_room.startswith('r'):
                # Robots are 100% reliable, .999 used to avoid 0 probabilities
                p = 0.999
                if num_pp > 0:
                    actions_dict['lights'+seen_room.split('r')[1]]='on'
                    state[seen_room]['table']=odict([((False,), 1 - p) ,((True,), p),])
                else:
                    actions_dict['lights'+seen_room.split('r')[1]]='off'
                    state[seen_room]['table']=odict([((False,), p) ,((True,), 1 - p),])



       #actions_dict = {'lights1': 'off', 'lights2': 'off', 'lights3': 'off', 'lights4': 'off', 'lights5': 'off', 'lights6': 'off', 'lights7': 'off', 'lights8': 'off', 'lights9': 'off', 'lights10': 'off', 'lights11': 'off', 'lights12': 'off', 'lights13': 'off', 'lights14': 'off', 'lights15': 'off', 'lights16': 'off', 'lights17': 'off', 'lights18': 'off', 'lights19': 'off', 'lights20': 'off', 'lights21': 'off', 'lights22': 'off', 'lights23': 'off', 'lights24': 'off', 'lights25': 'off', 'lights26': 'off', 'lights27': 'off', 'lights28': 'off', 'lights29': 'off', 'lights30': 'off', 'lights31': 'off', 'lights32': 'off', 'lights33': 'off', 'lights34': 'off', 'lights35':'off'}
    previous_state=state.copy()

    return actions_dict

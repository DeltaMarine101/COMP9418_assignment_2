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
    'r1' : ['r2', 'r3'],
    'r2' : ['r1', 'r4'],
    'r3' : ['r1', 'r7'],
    'r4' : ['r2', 'r8'],
    'r5' : ['r6', 'r9', 'c3'],
    'r6' : ['r5', 'c3'],
    'r7' : ['r3', 'c1'],
    'r8' : ['r4', 'r9'],
    'r9' : ['r5', 'r8', 'r13'],
    'r10': ['c3'],
    'r11': ['c3'],
    'r12': ['r22', 'outside'],
    'r13': ['r9', 'r24'],
    'r14': ['r24'],
    'r15': ['c3'],
    'r16': ['c3'],
    'r17': ['c3'],
    'r18': ['c3'],
    'r19': ['c3'],
    'r20': ['c3'],
    'r21': ['c3'],
    'r22': ['r12', 'r25'],
    'r23': ['r24'],
    'r24': ['r13', 'r14', 'r23'],
    'r25': ['r22', 'r26'],
    'r26': ['r25', 'r27'],
    'r27': ['r26', 'r32'],
    'r28': ['c4'],
    'r29': ['r30', 'c4'],
    'r30': ['r29'],
    'r31': ['r32'],
    'r32': ['r27', 'r31', 'r33'],
    'r33': ['r32'],
    'r34': ['c2'],
    'r35': ['c4'],
    'c1' : ['r7', 'r25', 'c2'],
    'c2' : ['r34', 'c1', 'c4'],
    'c3' : ['r5', 'r6', 'r10', 'r11', 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'o1'],
    'c4' : ['r28', 'r29', 'r35', 'c2', 'o1'],
    'o1' : ['c3', 'c4'],
    'outside': ['r12']
}

# This just adds itself to all rooms
for i in previous_G.keys():
    previous_G[i] += [i]

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

#Learn the outcomespace

def learn_outcome_space(data):
    outcomeSpace=dict()
    for i in data.keys():
        outcomeSpace[i] = tuple(np.unique(data[i]))
        # previous timestep nodes
        outcomeSpace[i + 't'] = outcomeSpace[i]
    return outcomeSpace

data = pd.read_csv('data.csv')
data_numpy = data.to_numpy()
data_cols = list(data.columns)
#first column is index in excel
del data[data.columns[0]]

#We are going to consider just if a room is empty or not (we don't care about the exact number of people)
room_columns=[]
for i in range(1,36):
    room_columns.append('r'+str(i))


all_rooms = list(previous_G.keys())
# print(all_rooms)

data_processed = {}
#room_columns=tuple(room_columns)
for i in all_rooms + list(door_sens_loc.keys()):
    data_processed[i] = (data_numpy[:,data_cols.index(i)] > 0)
    # data[i]=data[i].replace(to_replace=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],value=1)
    # data[i].where(~(data.i!=0),other=1,inplace=True)

# Might need to do something if "None" is returned
for i in list(urel_sens_loc.keys()) + list(rel_sens_loc.keys()):
    data_processed[i] = (data_numpy[:,data_cols.index(i)] == "motion")

outcomeSpace = learn_outcome_space(data_processed)

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

    # 'r16t' Denotes previous time as opposed to current time 'r16'
    if parent_offest: parent_names = [i + 't' for i in parent_names]

    return {'dom': tuple(list(parent_names)+[var_name]), 'table': prob_table}

#function from tutorial to calculate probability given an entry
def prob(factor, *entry):
    return factor['table'][entry]

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

#Function from tutorial to marginalize a variable from a factor
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

#Function from tutorial to make queries, the only difference is that this function does not normalize at
#the end. Given that we are classifying, it is just needed to choose the most likely one. In addition, we are
#assuming that q_vars is going to be just one variable because we are doing classification.
def query(p, outcomeSpace, q_vars, **q_evi):
    """
    argument
    `p`, probability table to query.
    `outcomeSpace`, dictionary will variable domains
    `q_vars`, list of variables in query head
    `q_evi`, dictionary of evidence in the form of variables names and values

    Returns a new factor  with all hidden variables eliminated as evidence set as in q_evi
    """

    # Let's make a copy of these structures, since we will reuse the variable names
    pm = p.copy()
    outSpace = {}
    for i in outcomeSpace.keys():
        if i in p['dom']: outSpace[i] = outcomeSpace[i]

    # First, we set the evidence
    for var_evi, e in q_evi.items():
        outSpace = evidence(var_evi, e, outSpace)

    # Second, we eliminate hidden variables NOT in the query
    # print("%%", outSpace)
    for var in outSpace:
        if var != q_vars:                         #this is different to the tutorial, we are assuming just one var
            pm = marginalize(pm, var, outSpace)
    return pm

# Get all transition probabilities
tran_prob_table=odict()
for present, previous in previous_G.items():
    tran_prob_table[present] = estProbs(data_processed, present, previous, outcomeSpace, parent_offest=1)

emis_prob_table=odict()
for sensor, location in all_sensors.items():
    emis_prob_table[sensor] = estProbs(data_processed, sensor, location, outcomeSpace)

# for k in emis_prob_table.keys():
#     print("Table for", k)
#     printFactor(emis_prob_table[k])
#     print()

# for k in tran_prob_table.keys():
#     print("Table for", k)
#     printFactor(tran_prob_table[k])
#     print()

# Initial state
state = {
    'r1' : 0, 'r2' : 0, 'r3' : 0, 'r4' : 0, 'r5' : 0, 'r6' : 0, 'r7' : 0, 'r8' : 0, 'r9' : 0,
    'r10': 0, 'r11': 0, 'r12': 0, 'r13': 0, 'r14': 0, 'r15': 0, 'r16': 0, 'r17': 0, 'r18': 0,
    'r19': 0, 'r20': 0, 'r21': 0, 'r22': 0, 'r23': 0, 'r24': 0, 'r25': 0, 'r26': 0, 'r27': 0,
    'r28': 0, 'r29': 0, 'r30': 0, 'r31': 0, 'r32': 0, 'r33': 0, 'r34': 0, 'r35': 0,
    'c1' : 0, 'c2' : 0, 'c3' : 0, 'c4' : 0,
    'o1' : 0,
    'outside': 1
}

prev_sens_data = None # might not be needed

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global tran_prob_table
    global emis_prob_table
    global prev_sens_data
    global state
    global previous_G

    cuttoff = 0.5 # TODO improve cutoff

    # TODO: Add code to generate your chosen actions, using the current state and sensor_data
    new_state = {
        'r1' : 0, 'r2' : 0, 'r3' : 0, 'r4' : 0, 'r5' : 0, 'r6' : 0, 'r7' : 0, 'r8' : 0, 'r9' : 0,
        'r10': 0, 'r11': 0, 'r12': 0, 'r13': 0, 'r14': 0, 'r15': 0, 'r16': 0, 'r17': 0, 'r18': 0,
        'r19': 0, 'r20': 0, 'r21': 0, 'r22': 0, 'r23': 0, 'r24': 0, 'r25': 0, 'r26': 0, 'r27': 0,
        'r28': 0, 'r29': 0, 'r30': 0, 'r31': 0, 'r32': 0, 'r33': 0, 'r34': 0, 'r35': 0,
        'c1' : 0, 'c2' : 0, 'c3' : 0, 'c4' : 0,
        'o1' : 0,
        'outside': 0
    }

    for i in state.keys():
        if prev_sens_data:
            table_index = tuple([(0, 1)[state[j] >= cuttoff ] for j in previous_G[i]])
            # Dealing with borken sensors
            # (TODO: Not sure if they pass in None or 'None'... Need to double check)
            if i in sens_rooms.keys() and sensor_data[sens_rooms[i]] != None:
                # print("IF", i)
                sensor = sens_rooms[i]
                # printFactor(tran_prob_table[i])
                joint = join(tran_prob_table[i], emis_prob_table[sensor], outcomeSpace)
                # tab = marginalize(joint, i, outcomeSpace)
                # printFactor(joint)
                evidence = {}
                evidence[sensor] = (0, 1)[sensor_data[sensor] == 'motion']
                # print(joint['dom'], "###")
                proba = query(joint, outcomeSpace, i, **evidence)['table']
                proba_t = proba[(1,)]
                proba_f = proba[(0,)]
                proba = proba_t / (proba_t + proba_f)
                # print("Probability true:", proba)
                # add second room information?
                # add sensor info
                new_state[i] = proba
            else:
                # print("ELSE", i)
                # print(tran_prob_table[i]['dom'], "###")
                evidence = {}
                proba = query(tran_prob_table[i], outcomeSpace, i, **evidence)['table']
                proba_t = proba[(1,)]
                proba_f = proba[(0,)]
                proba = proba_t / (proba_t + proba_f)
                # print("Probability true:", proba)
                new_state[i] = proba

    for i in ['robot1','robot2']:
        #info=sensor_data[i]
        if sensor_data[i] != None:
            seen_room=sensor_data[i].split(',')[0].partition("'")[2].partition("'")[0]
            num_pp=int(sensor_data[i].split(',')[1].strip().partition(')')[0])
            if seen_room.startswith('r'):
                if num_pp > 0:
                    # actions_dict['lights'+seen_room.split('r')[1]]='on'
                    new_state[seen_room] = 1.0
                else:
                    # actions_dict['lights'+seen_room.split('r')[1]]='off'
                    new_state[seen_room] = 0.0



    prev_sens_data = sensor_data
    state = new_state


    actions_dict = {'lights1': 'off', 'lights2': 'off', 'lights3': 'off', 'lights4': 'off', 'lights5': 'off', 'lights6': 'off', 'lights7': 'off', 'lights8': 'off', 'lights9': 'off', 'lights10': 'off', 'lights11': 'off', 'lights12': 'off', 'lights13': 'off', 'lights14': 'off', 'lights15': 'off', 'lights16': 'off', 'lights17': 'off', 'lights18': 'off', 'lights19': 'off', 'lights20': 'off', 'lights21': 'off', 'lights22': 'off', 'lights23': 'off', 'lights24': 'off', 'lights25': 'off', 'lights26': 'off', 'lights27': 'off', 'lights28': 'off', 'lights29': 'off', 'lights30': 'off', 'lights31': 'off', 'lights32': 'off', 'lights33': 'off', 'lights34': 'off', 'lights35':'off'}
    for i in state.keys():
        if i[0] == 'r':
            actions_dict['lights' + i[1:]] = ('off', 'on')[state[i] >= cuttoff]

    return actions_dict

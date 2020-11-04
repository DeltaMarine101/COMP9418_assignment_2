'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: Jeremie Kull    zID: 5208518

Name:     zID:
'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re


###################################
# Code stub
#
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
#

# Office structure
office_G = {
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
    'c2' : ['r34', 'c1'],
    'c3' : ['r5', 'r6', 'r10', 'r11', 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'o1'],
    'c4' : ['r28', 'r29', 'c2', 'o1'],
    'o1' : ['c3', 'c4'],
    'outside': ['r12']
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

# this global state variable demonstrates how to keep track of information over multiple
# calls to get_action

# initial probability distribution
state = {
    'r1' : 0, 'r2' : 0, 'r3' : 0, 'r4' : 0, 'r5' : 0, 'r6' : 0, 'r7' : 0, 'r8' : 0, 'r9' : 0,
    'r10': 0, 'r11': 0, 'r12': 0, 'r13': 0, 'r14': 0, 'r15': 0, 'r16': 0, 'r17': 0, 'r18': 0,
    'r19': 0, 'r20': 0, 'r21': 0, 'r22': 0, 'r23': 0, 'r24': 0, 'r25': 0, 'r26': 0, 'r27': 0,
    'r28': 0, 'r29': 0, 'r30': 0, 'r31': 0, 'r32': 0, 'r33': 0, 'r34': 0, 'r35': 0,
    'c1' : 0, 'c2' : 0, 'c3' : 0, 'c4' : 0,
    'o1' : 0,
    'outside': 1
}
training_data = pd.read_csv('data.csv')

# Pre-processing raw data into probability tables
# TODO: create prob tables

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global state
    global training_data

    # TODO: Add code to generate your chosen actions, using the current state and sensor_data

    actions_dict = {'lights1': 'off', 'lights2': 'off', 'lights3': 'off', 'lights4': 'off', 'lights5': 'off', 'lights6': 'off', 'lights7': 'off', 'lights8': 'off', 'lights9': 'off', 'lights10': 'off', 'lights11': 'off', 'lights12': 'off', 'lights13': 'off', 'lights14': 'off', 'lights15': 'off', 'lights16': 'off', 'lights17': 'off', 'lights18': 'off', 'lights19': 'off', 'lights20': 'off', 'lights21': 'off', 'lights22': 'off', 'lights23': 'off', 'lights24': 'off', 'lights25': 'off', 'lights26': 'off', 'lights27': 'off', 'lights28': 'off', 'lights29': 'off', 'lights30': 'off', 'lights31': 'off', 'lights32': 'off', 'lights33': 'off', 'lights34': 'off', 'lights35':'off'}
    for i in actions_dict.keys():
        actions_dict[i] = 'on'
    return actions_dict

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 01:02:10 2020

@author: pablopacheco
"""

#we are going to use log probs
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
            #s = s + p                            # Sum over all values of var by accumulating the sum in s.
            #we are going to use log probs
            if p<0:
                s=s+math.exp(p)
            else:
                s=s+p
        # Create a new table entry with the multiplication of p1 and p2
        table.append((entries, s))
    return {'dom': tuple(new_dom), 'table': odict(table)}


#function from tutoral to calculate join probability given two factors, except this use log probs
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
        if p1 >0:
            p1=math.log(p1)
        elif p1==0:             ### WE DON'T KNOW IF WE ARE RECEIVING A NORMAL PROB OR A LOG PROB
            p1=-1000000
        p2 = prob(f2, *f2_entry)
        if p2 >0:
            p2=math.log(p2)
        elif p1==0:             ### WE DON'T KNOW IF WE ARE RECEIVING A NORMAL PROB OR A LOG PROB
            p2=-1000000

        # Create a new table entry with the multiplication of p1 and p2
        table.append((entries, p1 + p2))
    return {'dom': tuple(common_vars), 'table': odict(table)}


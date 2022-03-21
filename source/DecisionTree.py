from typing import Dict
import pandas as pd
import math

def get_probability(dataframe: pd.DataFrame, feature):
    p_ret = dict()
    items = pd.unique(dataframe[feature])
    n_total = dataframe[feature].count()
    for item in items:
        ni = dataframe[feature].loc[dataframe[feature] == item].count()
        pi = ni/n_total
        p_ret[item] = pi
    return p_ret

def get_entropy(dataframe: pd.DataFrame, feature):
    # Get all values in the label 
    items = pd.unique(dataframe[feature])
    n_total = dataframe[feature].count()
    entropy = 0
    # Get p for all values possible --> Entropy_s
    for item in items:
        n_i = dataframe[feature].loc[dataframe[feature] == item].count()
        pi = n_i/n_total
        entropy -= pi*math.log2(pi)
    return entropy

def get_gain(dataframe: pd.DataFrame, label, feature):
    # Get all values in the label 
    items = pd.unique(dataframe[label])
    n_total = dataframe[label].count()
    entropy_s = 0
    # Get p for all values possible --> Entropy_s
    for item in items:
        n_i = dataframe[label].loc[dataframe[label] == item].count()
        pi = n_i/n_total
        entropy_s -= pi*math.log2(pi)
    # Entropy of feature
    ## Get all the unique values of the feature 
    values = pd.unique(dataframe[feature])
    entropy_feature = 0
    ### Number of total
    n_feature = dataframe[feature].count()
    p_feature = dict()
    values = pd.unique(dataframe[feature])
    entropy_feature = 0
    for item in values:
        n_subtotal = dataframe[feature].loc[dataframe[feature] == item].count()
        p_feature[item] = n_subtotal/n_feature
        subdat = dataframe.loc[dataframe[feature] == item]
        subitems = pd.unique(subdat[label])
        entropy_sub = 0
        for sub in subitems:
            ni = subdat[label].loc[subdat[label] == sub].count()
            pi = ni/n_subtotal
            entropy_i = -pi*math.log2(pi)
            entropy_sub += entropy_i
        entropy_feature += p_feature[item]*entropy_sub
    gain = entropy_s - entropy_feature
    return gain

def get_fmax(dat: pd.DataFrame, label: str):
    features = dat.drop(label, axis=1).keys()
    fmax = features[0]
    gmax = get_gain(dat, label, fmax)
    for fi in features:
        gi = get_gain(dat, label, fi)
        if gmax < gi: gmax, fmax = gi, fi
    return fmax

def build_decision_tree(dat: pd.DataFrame, label: str):
    decisiontree: Dict = dict()
    decision: Dict = dict()
    fmax = get_fmax(dat, label)
    subvalues = pd.unique(dat[fmax])
    # All items searching
    for item in subvalues:
        # print(f'The sub-table with {fmax} = {item}')
        subdummy = dat.loc[dat[fmax] == item]
        # print(subdummy)
        entropy_sub = get_entropy(subdummy, label)
        if entropy_sub == 0:
            decision[item] = pd.unique(subdummy[label])[0]
        else:
            # print('Processing subtree')
            decision[item] = build_decision_tree(subdummy.drop(fmax, axis=1), label)
        # print(f'Entropy = {entropy_sub}\n')
        decisiontree[fmax] = decision
    return decisiontree

__all__ = ['get_gain', 'get_probability', 'get_entropy', 'build_decision_tree']
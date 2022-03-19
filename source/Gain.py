import pandas as pd
import math

def get_gain(dataframe, label, feature):
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

__all__ = ['get_gain']
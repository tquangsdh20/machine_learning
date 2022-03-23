from typing import List
from pandas import DataFrame, Categorical, unique
from imblearn.over_sampling import SMOTE


def categorical_to_numberic(dat: DataFrame, features: List[str]):
    for feature in features:
        dat[feature] = Categorical(dat[feature])
        values = unique(dat[feature])
        dat[feature].replace(
            values,
            list(range(len(values))),
            inplace=True
        )
    return dat

def smote(dat: DataFrame, label: str):
    """The features must be numberic"""
    oversample = SMOTE()
    features = dat.drop(label, axis=1).keys()
    dat_smote, dat_label = oversample.fit_resample(dat[features], dat[label])
    return dat_smote, dat_label

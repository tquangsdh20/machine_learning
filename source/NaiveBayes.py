from sklearn.naive_bayes import GaussianNB

def naive_bayes(dat, label):
    model = GaussianNB()
    # fit it to training data
    model.fit(dat, label)
    return model



import numpy as np


def calculate_maxlen(docs, percentile=90):
    return int(np.percentile(np.array([len(doc) for doc in docs]), percentile))
import numpy as np
import statistics as stat

def kfold(classifier, X, y, k=5, random_seed=-1):
    """Execute k-fold cross-validation on data provided with specified classifier model.

    Parameters:
    classifier - classifier model
    X - array with input samples, shape (n_samples, n_features) 
    Y - array with target values, corresponding to input samples, shape (n_samples,)
    k - number of folds data will be partioned into
    random_seed - random seed for data partitioning

    Return:
    scores - list of scores for model predictions on all k folds
    mean - mean of all scores
    variance - variance of all scores
    """
    if (random_seed > 0):
        np.random.seed(random_seed)
    scores = []

    ids = np.random.permutation(X.shape[0])
    parts_arrays = np.array_split(ids, k)
    parts = []
    for el in parts_arrays:
        parts.append(list(el))

    for i, test_ids in enumerate(parts):
        train_ids = []
        for j in range(k):
            if j != i:
                train_ids += parts[j]
        classifier.fit(np.array(X.iloc[train_ids]), np.array(y.iloc[train_ids]))
        score = classifier.score(np.array(X.iloc[test_ids]), np.array(y.iloc[test_ids]))
        scores.append(score)
    
    return scores, stat.mean(scores), stat.variance(scores)    
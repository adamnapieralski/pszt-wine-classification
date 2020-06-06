import numpy as np
import statistics as stat

def kfold(classifier, X, y, k=5, random_seed=-1):
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
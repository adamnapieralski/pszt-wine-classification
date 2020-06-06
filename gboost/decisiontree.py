"""Decision Tree Classifier

Implements Decision Tree Classifier that is generated using ID3 algorithm.
"""
__author__ = "Napieralski Adam, Kostrzewa Lukasz"

import numpy as np 
from sklearn.datasets import load_iris

class Node():
    """Represents decision tree node.
    """
    def __init__(self, value=None, split_feature_index=None, feature_cutoff=None,
                 left_child=None, right_child=None):
        self.value = value
        self.split_feature_index = split_feature_index
        self.feature_cutoff = feature_cutoff
        self.left_child = left_child
        self.right_child = right_child

class DecisionTree():
    """
    Decision Tree Classifier implemented using ID3 algorithm. The algorithm uses
    inforamtion gain criterion. Its interface was implemented based on
    Scikit-learn Decision Trees.
    """

    def __init__(self, max_depth=4):
        self.depth = 0
        self.max_depth = max_depth
        self.tree_root = None
            
    def fit(self, X, Y):
        """Creates decision tree.
        """
        self.tree_root = self._build_decision_tree(X,Y)

    def predict(self, X):
        """Predicts categories for the given data.
        """
        if self.tree_root is None:
            print('Error, decision tree not trained')
            return
        Y = np.empty(X.shape[0])
        for ind, x in enumerate(X):
            Y[ind] = self._predict_single(x)
        return Y

    def _predict_single(self, x):
        """Predict category for the single data row.
        """
        node = self.tree_root
        while node.feature_cutoff is not None:
            if x[node.split_feature_index] < node.feature_cutoff:
                if node.left_child is not None:
                    node = node.left_child
                else:
                    return node.value
            else:
                if node.right_child is not None:
                    node = node.right_child
                else:
                    return node.value
        else:
            return node.value
        

    def _build_decision_tree(self, x, y, depth=0, parent_node = Node()):
        """Build next level of the decision tree.
        """

        # node can't be created
        if parent_node is None or y.size == 0 or depth > self.max_depth:
            return None

        # all values the same, no further recursion needed
        if np.all(y == y[0]):
            return Node(y[0])

        #split
        feature_index, feature_cutoff_value = self._find_best_split(x, y)
        
        node_value = np.mean(y)

        node = Node(value=node_value, split_feature_index=feature_index, feature_cutoff=feature_cutoff_value)

        #left child
        left_node_x = x[x[:, feature_index] < feature_cutoff_value]        
        left_node_y = y[x[:, feature_index] < feature_cutoff_value]        
        left_node_x = np.delete(left_node_x, feature_index, 1)
        if left_node_x.size > 0:
            node.left_child = self._build_decision_tree(left_node_x, left_node_y, depth+1) 
        else:
            node.left_child = None


        #right child
        right_node_x = x[x[:, feature_index] >= feature_cutoff_value]        
        right_node_y = y[x[:, feature_index] >= feature_cutoff_value]        
        right_node_x = np.delete(right_node_x, feature_index, 1)
        if right_node_x.size > 0:
            node.right_child = self._build_decision_tree(right_node_x, right_node_y, depth+1)
        else:
            node.right_child = None
              

        self.depth += 1
        return node
        

    def _entropy(self, S):
        """Entropy of the given set of values    
        """
        occurances_count = np.unique(S, return_counts=True)[1] / S.size   
        return -1 * np.sum(occurances_count * np.log(occurances_count)) 

    def _subsets_entropy(self, sets):
        """
        For each subset of the given set computes entropy
        and returns their weighted sum    
        """
        H = 0 
        size = sets.size
        for s in sets:
            H += s.size / size * self._entropy(s)
        return H

    def _find_best_split_for_feature(self, feature, target):
        """
        Finds the split value in the feature array that minimizes
        the entropy computed using target array.
        
        Example:
        let feature = [1,2,3,4,5]
        let target = [1,1,1,3,3]
        the function returns the value 4 and the weighted sum of entropies
        for [1, 1, 1] and [3, 3] subsets
        """               
        
        f = np.unique(feature)

        min_entropy = 10000000000    
        cutoff_value = f[0]

        for value in f:            
            group_1 = target[feature < value]
            group_2 = target[feature >= value]
            entropy = self._subsets_entropy(np.array([group_1, group_2]))
            if entropy < min_entropy:
                min_entropy = entropy
                cutoff_value = value
        return min_entropy, cutoff_value


    def _find_best_split(self, X, Y):
        """
        Finds the best split value.
        returns:
        the index of the best fature to split
        the best cutoff value        
        """
        best_feature_index = None
        min_entropy = 10000000000
        best_feature_cutoff_value = None
        for feature_ind, feature in enumerate(X.T):
            entropy, cutoff_value = self._find_best_split_for_feature(feature, Y)
            if entropy == 0:
                return feature_ind, cutoff_value
            elif entropy < min_entropy:
                min_entropy = entropy
                best_feature_index = feature_ind
                best_feature_cutoff_value = cutoff_value
        return best_feature_index, best_feature_cutoff_value 



if __name__ == "__main__":    
    """Run test using iris dataset.
    """

    iris = load_iris()

    x = iris.data
    y = iris.target

    tree = DecisionTree(max_depth=7)
    tree.fit(x, y)
    yy = tree.predict(x)
    
    correct_predictions = 0
    for i in range(y.shape[0]):
        if y[i] == round(yy[i]):
            correct_predictions += 1
        print(y[i], round(yy[i]))
    print('results: ', correct_predictions / y.shape[0] * 100, '%')

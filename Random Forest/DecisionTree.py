import numpy as np  # Add this line
from collections import Counter

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree():
    def __init__(self, min_splits=2, max_depth=100, n_features=None):
        self.min_splits = min_splits
        self.max_depth = max_depth
        self.root = None
        self.n_features = n_features

    def fit(self, x, y):
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1], self.n_features)
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        n_samples, n_feature = x.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_splits:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = np.random.choice(n_feature, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(x, y, feature_indices)
        left, right = self._split(x[:, best_feature], best_threshold)
        left = self._grow_tree(x[left, :], y[left], depth + 1)
        right = self._grow_tree(x[right, :], y[right], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, x, y, feature_indices):
        gain = -1
        split_index, split_threshold = None, None
        for feature in feature_indices:
            x_col = x[:, feature]
            thresholds = np.unique(x_col)
            for t in thresholds:
                best = self._info_gain(x_col, y, t)
                if gain < best:
                    gain = best
                    split_index = feature
                    split_threshold = t
        return split_index, split_threshold

    def _info_gain(self, x, y, threshold):
        parent_entropy = self._entropy(y)
        left, right = self._split(x, threshold)
        if len(left) == 0 or len(right) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = self._entropy(y[left]), self._entropy(y[right])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, x, threshold):
        left = np.argwhere(x <= threshold).flatten()
        right = np.argwhere(x > threshold).flatten()
        return left, right

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

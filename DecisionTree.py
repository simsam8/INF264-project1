import numpy as np
import pandas as pd
import collections
from Node import Node


class DecisionTree:
    """
    Class which implements the decision tree algorithm
    """

    def __init__(self) -> None:
        # self.min_samples_split = min_samples_split
        # self.max_depth = max_depth
        # self.n_features = n_features
        self.root_node: Node

    def _equal_feature_values(self, X) -> bool:
        """
        Checks if all features have equal values in their rows

        Params
        ----------
        X: feature colums

        return: bool
        """
        equal_row_values = np.all(X == X[0, :], axis=0)

        for equal in equal_row_values:
            if not equal:
                return False
        return True

    def _calculate_entropy(self, y) -> np.floating:
        """
        Helper function for calculating entropy

        Params
        ----------
        y: labels

        return: entropy
        """
        probablities = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in probablities if p > 0])

    def _calculate_gini(self, y) -> np.floating:
        """
        Helper function for calculating gini index

        Params
        ----------
        y: labels

        return: gini index
        """
        probabilities = np.bincount(y) / len(y)
        return 1 - np.sum(probabilities**2)

    def _information_gain(
        self, y, X_column, threshold, impurity_measure
    ) -> np.floating | int:
        """
        Calculate information gain for a feature column


        Params
        ----------
        y: labels
        X_column: feature column
        threshold: feature value to split from
        impurity_measure: "entropy" or "gini"

        return: information gain
        """
        # Set impurity function from impurity measure
        if impurity_measure == "entropy":
            impurity_func = self._calculate_entropy
        elif impurity_measure == "gini":
            impurity_func = self._calculate_gini
        else:
            raise Exception(
                f"There is no impurity function for measure: {impurity_measure}"
            )

        impurity_parent = impurity_func(y)

        # get children

        left_indexes, right_indexes = self._split(X_column, threshold)

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0

        # calculate impurity on children

        fraction_left = len(left_indexes) / len(y)
        fraction_right = len(right_indexes) / len(y)

        impurity_left = impurity_func(y[left_indexes])
        impurity_right = impurity_func(y[right_indexes])

        # Calculate information gain

        impurity_child = fraction_left * impurity_left + fraction_right * impurity_right

        information_gain = impurity_parent - impurity_child
        return information_gain

    def learn(self, X, y, impurity_measure="entropy") -> None:
        """
        Learn a decision tree from training features and labels

        Params
        ----------
        X: feature columns
        y: labels
        impurity_measure: "entropy" or "gini"
        """
        # Convert DataFrame to nparray
        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        # Convert Series to nparray
        if type(y) == pd.Series:
            y = y.to_numpy()

        self.root_node = self._build_tree(X, y, impurity_measure)

    def _split(self, X_column, threshold):
        """
        Splits feature column on a threshold

        Params
        ----------
        X_column: feature column
        threshold: threshold to split on

        return: (left, right) indexes
        """
        left_indexes = np.argwhere(X_column <= threshold).flatten()
        right_indexes = np.argwhere(X_column > threshold).flatten()

        return left_indexes, right_indexes

    def _best_split(self, X, y, feature_idexes, impurity_measure):
        """
        Gets the best split based on impurity_measure

        Params
        ----------
        X: feature columns
        y: labels
        feature_indexes: feature indexes
        impurity_measure: "entropy" or "gini"

        return: (index, threshold) of split
        """
        best_gain = -1
        split_index, split_threshold = None, None

        for feature_index in feature_idexes:
            X_column = X[:, feature_index]

            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # calc information gain
                if_gain = self._information_gain(
                    y, X_column, threshold, impurity_measure
                )

                if if_gain > best_gain:
                    best_gain = if_gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def _majority_label(self, y):
        """
        Gets the most common label

        Params
        ----------
        y: labels

        return: most common label
        """
        counts = collections.Counter(y)
        label = counts.most_common(1)[0][0]
        return label

    def _build_tree(self, X, y, impurity_measure, level=0) -> Node:
        """
        Recursive helper function to build the decision tree

        Params
        ----------
        X: feature columns
        y: labels
        impurity_measure: "entropy" or "gini"
        level: tree depth

        return: leaf node or decision node
        """
        # X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check for stopping criteria
        if (
            n_labels
            == 1
            # or level >= self.max_depth
            # or n_samples < self.min_samples_split
        ):
            leaf_value = self._majority_label(y)
            node = Node(value=leaf_value)
            return node

        elif self._equal_feature_values(X):
            leaf_value = self._majority_label(y)
            return Node(value=leaf_value)

        feature_indexes = list(range(n_features))
        # feature_indexes = np.random.choice(n_features, self.n_features, replace=False)

        # best split

        best_feature, best_threshold = self._best_split(
            X, y, feature_indexes, impurity_measure
        )

        # create children
        left_indexes, right_indexes = self._split(X[:, best_feature], best_threshold)

        left = self._build_tree(
            X[left_indexes, :], y[left_indexes], impurity_measure, level + 1
        )
        right = self._build_tree(
            X[right_indexes, :], y[right_indexes], impurity_measure, level + 1
        )
        return Node(best_feature, best_threshold, left, right)

    def _predict(self, x, node: Node):
        """
        Recursive helper function to traverse tree
        """
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict(x, node.data_left)
        return self._predict(x, node.data_right)

    def display_tree(self):
        """
        Display the decision tree
        """
        self._display(self.root_node)

    def _display(self, node: Node, level=0):
        """
        Recursive helper function for traversing tree
        """
        if node.is_leaf():
            node.display(level)

        else:
            node.display(level)

            self._display(node.data_left, level + 1)
            self._display(node.data_right, level + 1)

    def predict(self, X):
        """
        Prediction function for classifying new data

        Params
        ----------
        X: feature columns

        return: numpy array of prediction(s)
        """
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        return np.array([self._predict(x, self.root_node) for x in X])


if __name__ == "__main__":
    pass

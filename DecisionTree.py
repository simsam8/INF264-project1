import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from copy import deepcopy
from Node import Node


class DecisionTree:
    """
    Class which implements the decision tree algorithm
    """

    def __init__(self, random_state=None) -> None:
        self.root_node: Node
        self.random_state = random_state

    def _equal_feature_values(self, X) -> bool:
        """
        Checks if all features have equal values in their rows

        Params
        ----------
        X: feature colums

        return: bool
        """
        equal_row_values = np.all(X == X[0, :], axis=0)

        return all(equal for equal in equal_row_values)

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

    def set_impurity_function(self, impurity_measure: str) -> None:
        """
        Sets impurity function of decision tree

        Params
        ----------
        impurity_measure: "entropy" or "gini"

        return: None
        """
        if impurity_measure == "entropy":
            self.impurity_function = self._calculate_entropy
        elif impurity_measure == "gini":
            self.impurity_function = self._calculate_gini
        else:
            raise Exception(f"There is no impurity function for: {impurity_measure}")

    def _information_gain(self, y, X_column, threshold) -> np.floating | int:
        """
        Calculate information gain for a feature column

        Params
        ----------
        y: labels
        X_column: feature column
        threshold: feature value to split from

        return: information gain
        """

        impurity_parent = self.impurity_function(y)

        # get children
        left_indexes, right_indexes = self._split(X_column, threshold)

        # check if either split is empty
        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0

        # calculate impurity on children
        fraction_left = len(left_indexes) / len(y)
        fraction_right = len(right_indexes) / len(y)

        impurity_left = self.impurity_function(y[left_indexes])
        impurity_right = self.impurity_function(y[right_indexes])

        # Calculate information gain
        impurity_child = fraction_left * impurity_left + fraction_right * impurity_right

        information_gain = impurity_parent - impurity_child
        return information_gain

    def _split(self, X_column, threshold) -> tuple:
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

    def _best_split(self, X, y, feature_idexes) -> tuple:
        """
        Gets the best split based on impurity_measure

        Params
        ----------
        X: feature columns
        y: labels
        feature_indexes: feature indexes

        return: (index, threshold) of split
        """
        best_gain = -1
        split_index, split_threshold = None, None

        for feature_index in feature_idexes:
            X_column = X[:, feature_index]

            thresholds = np.unique(X_column)

            for threshold in thresholds:
                if_gain = self._information_gain(y, X_column, threshold)

                if if_gain > best_gain:
                    best_gain = if_gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def _majority_label(self, y) -> int:
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

    def learn(self, X, y, impurity_measure="entropy", prune=False) -> None:
        """
        Learn a decision tree from training features and labels

        Params
        ----------
        X: feature columns
        y: labels
        impurity_measure: "entropy" or "gini"
        """
        self.set_impurity_function(impurity_measure)

        # Convert DataFrame to nparray
        if type(X) == pd.DataFrame:
            X = X.to_numpy()

        # Convert Series to nparray
        if type(y) == pd.Series:
            y = y.to_numpy()

        if prune:
            (
                self.X_train,
                self.X_pruning,
                self.y_train,
                self.y_pruning,
            ) = train_test_split(X, y, test_size=0.15, random_state=self.random_state)
        else:
            self.X_train, self.y_train = X, y

        self.root_node = self._build_tree(self.X_train, self.y_train)

        if prune:
            self.prune()

    def prune(self) -> None:
        """
        Prunes the decision tree and sets the root to the pruned tree

        return: None
        """
        self.pruned_tree = deepcopy(self.root_node)
        visited_nodes = set()
        self._prune(visited_nodes, self.pruned_tree)
        self.root_node = self.pruned_tree

    def _worse_accuracy(self) -> bool:
        """
        Compares the accuracy between pruned and unpruned tree

        return: bool
        """
        unpruned_pred = self.predict(self.X_pruning)
        pruned_pred = self._predict_pruned(self.pruned_tree, self.X_pruning)

        accuracy_unpruned = accuracy_score(self.y_pruning, unpruned_pred)
        accuracy_pruned = accuracy_score(self.y_pruning, pruned_pred)

        if accuracy_pruned >= accuracy_unpruned:
            return False
        else:
            return True

    def _prune(self, visited: set[Node], node: Node, prev_node: Node = None) -> None:
        """
        Recursively prune the decision tree, and compare with unpruned tree

        Params
        ----------
        visited: set of Nodes
        node: current node
        prev_node: previous node

        return: None
        """
        if node is None:
            pass
        elif node not in visited:
            visited.add(node)

            if node.is_leaf():
                feat = prev_node.feature
                thresh = prev_node.threshold
                left = prev_node.data_left
                right = prev_node.data_right
                prev_node.convert_to_leaf()

                if self._worse_accuracy():
                    # restore decision node and keep searching
                    prev_node.restore_decision_node(feat, thresh, left, right)
                    self._prune(visited, prev_node)
                else:
                    visited.add(right)
                    self._prune(visited, prev_node)

            else:
                self._prune(visited, node.data_left, node)
                self._prune(visited, node.data_right, node)

    def _build_tree(self, X, y, depth=0) -> Node:
        """
        Recursive helper function to build the decision tree

        Params
        ----------
        X: feature columns
        y: labels
        level: tree depth

        return: leaf node or decision node
        """
        n_features = X.shape[1]
        n_labels = len(np.unique(y))
        majority_label = self._majority_label(y)

        # Check for stopping criteria
        if n_labels == 1:
            return Node(value=majority_label)

        elif self._equal_feature_values(X):
            return Node(value=majority_label)

        feature_indexes = list(range(n_features))

        # best split
        best_feature, best_threshold = self._best_split(X, y, feature_indexes)

        # create children
        left_indexes, right_indexes = self._split(X[:, best_feature], best_threshold)

        left = self._build_tree(X[left_indexes, :], y[left_indexes], depth + 1)
        right = self._build_tree(X[right_indexes, :], y[right_indexes], depth + 1)

        return Node(best_feature, best_threshold, left, right, majority_label)

    def _predict(self, x, node: Node):
        """
        Recursive helper function to traverse tree

        Params
        ----------
        node: current node

        return: int or _predict
        """
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict(x, node.data_left)
        return self._predict(x, node.data_right)

    def display_tree(self) -> None:
        """
        Display the decision tree

        return: None
        """
        levels = self._display(self.root_node, set())
        print(f"There are {len(levels)} levels in tree.")

    def _display(self, node: Node, levels: set, level=0) -> set[int]:
        """
        Recursive helper function for traversing tree

        Params
        ----------
        node: current node
        levels: set of levels
        level: current level

        return: set of levels
        """
        levels.add(level)
        if node.is_leaf():
            node.display(level)

        else:
            node.display(level)

            self._display(node.data_left, levels, level + 1)
            self._display(node.data_right, levels, level + 1)
        return levels

    def predict(self, X) -> np.ndarray:
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

    def _predict_pruned(self, pruned_tree: Node, X) -> np.ndarray:
        """
        Prediction function for classifying new data on pruned tree

        Params
        ----------
        X: feature columns

        return: numpy array of prediction(s)
        """
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        return np.array([self._predict(x, pruned_tree) for x in X])


if __name__ == "__main__":
    pass

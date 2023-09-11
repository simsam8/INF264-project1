import numpy as np
import pandas as pd
from Node import Node


class DecisionTree:
    """
    Class which implements the decision tree algorithm
    """

    def __init__(self) -> None:
        self.root_node: Node

    def equal_features(self, X: pd.DataFrame) -> bool:
        # print(X)
        """
        Checks if all features has equal values.

        return -> bool
        """
        for col in X.columns:
            # if (X[col] == X[col].values[:1]).all():
            if len(X[col].unique()) == 1:
                pass
            else:
                return False
        return True

    def _calculate_entropy(self, y: pd.Series) -> float:
        """
        Helper function for calculating entropy

        return -> entropy: float
        """
        p = y.value_counts() / y.shape[0]
        return -np.sum(p * np.log2(p))

    def _calc_information_gain(
        self, parent: pd.Series, left: pd.Series, right: pd.Series
    ) -> float:
        """
        Helper function for calculating imformation gain.

        return -> information_gain: float
        """
        fraction_left = left.count() / parent.count()
        fraction_right = right.count() / parent.count()
        E_parent = self._calculate_entropy(parent)
        E_left = self._calculate_entropy(left)
        E_right = self._calculate_entropy(right)

        return E_parent - (fraction_left * E_left + fraction_right * E_right)

    # def calculate_information_gain(self, X: pd.DataFrame, y: pd.Series):
    #     """
    #     Returns the feature with the highest information gain
    #     """
    #     print("Dataset entropy")
    #     E_dataset = self.calculate_entropy(y)
    #     # print(E_dataset)
    #     information_gain = []
    #     for col in X.columns:
    #         bin_over = X[col].where(X[col] > X[col].mean()).dropna()
    #         bin_under = X[col].mask(X[col] > X[col].mean()).dropna()
    #         over_labels = y.loc[bin_over.index]
    #         under_labels = y.loc[bin_under.index]
    #         E_col = self.calculate_entropy(
    #             over_labels
    #         )  # + calculate_entropy(unde_labels)
    #         if_gain = E_dataset - E_col
    #         information_gain.append((col, if_gain))
    #         max_information_gain = max(information_gain)[0]
    #     return max_information_gain

    # def calculate_gini_index(self, X: pd.DataFrame, y: pd.Series):
    #     return 1 - np.sum(probability**2)

    def _best_split(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Helper function for finding the best split

        return -> dict =
        {
        feature: str,
        threshold: float,
        data_left: (Dataframe, Series),
        data_right: (Dataframe, Series),
        if_gain: float,
        }
        """
        best_split = {}
        best_if_gain = -1

        # Loop through every feature
        for col in X.columns:
            # Split dataset by mean of feature
        
            for threshold in X[col].unique():

                X_right = X.where(X[col] > threshold).dropna()
                X_left = X.mask(X[col] > threshold).dropna()
                y_right = y.loc[X_right.index]
                y_left = y.loc[X_left.index]

                # Calculate only if there is data in both splits
                if len(y_right) > 0 and len(y_right) > 0:
                    # Calculate information information gain
                    # Save split parameters if current split is better than previous
                    if_gain = self._calc_information_gain(y, y_left, y_right)
                    if if_gain > best_if_gain:
                        # print("x_right", X_right.shape)
                        # print("x_left", X_left.shape)
                        best_split = {
                            "feature": col,
                            "threshold": threshold,
                            "data_left": (X_left, y_left),
                            "data_right": (X_right, y_right),
                            "if_gain": if_gain,
                        }
                        best_if_gain = if_gain

        return best_split

    # def split_dataset_on_feature(self, X: pd.DataFrame, y: pd.Series, feature: str):
    #     split1 = X[feature].where(X[feature] > X[feature].mean()).dropna()
    #     split2 = X[feature].mask(X[feature] > X[feature].mean()).dropna()
    #     labels1 = y.loc[split1.index]
    #     labels2 = y.loc[split2.index]
    #     left_node = Node(split1, labels1)
    #     right_node = Node(split2, labels2)
    #     return left_node, right_node

    # def calculate_impurity(self, X: pd.DataFrame, y: pd.Series, type: str) -> str:
    #     if type == "entropy":
    #         return self.calculate_information_gain(X, y)
    #
    #     elif type == "gini":
    #         return self.calculate_gini_index(X, y)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, level) -> Node:
        # Check if it should be a leaf node

        # All labels are the same
        if len(y.unique()) == 1:
            leaf = Node(value=y.unique()[0])
            # leaf.display_leaf(level)
            # print("Leaf condition same labels")
            # print(y.unique(), leaf.value)
            # print()
            return leaf
        # All data points have identical feature values
        elif self.equal_features(X):
            leaf = Node(value=y.value_counts().max())
            # leaf.display_leaf(level)
            # print("Leaf condition same datapoints")
            # print(y.value_counts().max(), leaf.value)
            # print()
            return leaf
        else:
            best_split = self._best_split(X, y)

            if best_split == {}:
                pass
            # Check if split is not pure
            elif best_split["if_gain"] > 0:
                # Build branches
                # print("split")
                # print(best_split["data_left"])

                left_branch = self._build_tree(
                    X=best_split["data_left"][0],
                    y=best_split["data_left"][1],
                    level=level + 1,
                )

                right_branch = self._build_tree(
                    X=best_split["data_right"][0],
                    y=best_split["data_right"][1],
                    level=level + 1,
                )
                decision = Node(
                    feature=best_split["feature"],
                    threshold=best_split["threshold"],
                    data_left=left_branch,
                    data_right=right_branch,
                    gain=best_split["if_gain"],
                )

                # decision.display_decision(level)
                return decision
        test = Node(value=y.value_counts().max())
        print(y.value_counts())
        # print("Rest node")
        # print(test.value)
        return test

    def learn(self, X: pd.DataFrame, y: pd.Series, impurity_measure="entropy"):
        self.root_node = self._build_tree(X, y, 0)

    # def _display_tree(self, tree: Node, level):
    #     # Display leaf
    #     if tree.value is not None:
    #         tree.display_leaf(level+1)
    #
    #     tree.display_decision(level)
    #
    #
    #     return self._display_tree(level+1)

    def _predict(self, x: pd.DataFrame, tree: Node):
        """
        Recursive helper function
        """

        # Lmeaieaf node
        if tree.value is not None:
            return tree.value

        feature_value = x[1][tree.feature]
        # print(tree.feature, feature_value)

        # Go left
        if feature_value <= tree.threshold:
            # print(feature_value, " <= ", tree.threshold)
            return self._predict(x=x, tree=tree.data_left)

        # Go right
        if feature_value > tree.threshold:
            # print(feature_value, " > ", tree.threshold)
            return self._predict(x=x, tree=tree.data_right)

    def predict(self, x: pd.DataFrame):
        """
        Prediction function for classifying new data
        """

        return [self._predict(row, self.root_node) for row in x.iterrows()]


if __name__ == "__main__":
    pass

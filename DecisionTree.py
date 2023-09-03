import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self) -> None:
        pass

    def calculate_impurity(self, y: pd.Series, type: str) -> float:
        probability = y.value_counts() / y.shape[0]

        if type == "entropy":
            return np.sum(-probability*np.log2(probability))

        elif type == "gini":
            return 1 - np.sum(probability**2)

    def learn(self, X: np.ndarray, y: np.ndarray, impurity_measure: str):
        pass

    # Maybe parameter tree is not needed
    def predict(self, x: np.ndarray, tree):
        pass


if __name__ == "__main__":
    pass

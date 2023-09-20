import numpy as np
from pandas import DataFrame
from DecisionTree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import itertools
import operator


class Evaluation:
    """
    Class for evaluating differnt decision tree implementations
    """
    def __init__(
        self, X_train: DataFrame, y_train: DataFrame, random_state=None
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.seed = random_state

        impurity_measures = ["entropy", "gini"]
        pruning = [True, False]

        self.model_parameters = list(itertools.product(*[impurity_measures, pruning]))

    def best_model_of_implementation(self) -> tuple[DecisionTree, tuple]:
        """
        Evaluates my implementation of the DecisionTree
        with defined parameters and return the best model and parameters

        return: (model, parameters)
        """
        print("\nCross validation of implemented DecisionTree...")

        # Creating every possible model for evaluation
        models = [
            (params, DecisionTree(random_state=self.seed))
            for params in self.model_parameters
        ]

        # Evaluate each model
        model_results = []
        kfold = KFold(n_splits=10, random_state=self.seed, shuffle=True)

        for params, model in models:
            validation_scores = []

            for train_index, val_index in kfold.split(self.X_train, self.y_train):
                x_train_fold, x_val_fold = (
                    self.X_train.iloc[train_index],
                    self.X_train.iloc[val_index],
                )

                y_train_fold, y_val_fold = (
                    self.y_train.iloc[train_index],
                    self.y_train.iloc[val_index],
                )

                model.learn(x_train_fold, y_train_fold, params[0], params[1])
                val_prediction = model.predict(x_val_fold)
                val_score = accuracy_score(y_val_fold, val_prediction)
                validation_scores.append(val_score)

            mean_score = np.mean(validation_scores)
            model_results.append((model, mean_score))
            print(f"{params}: {mean_score}")

        # Test score on best model
        best_model_index = model_results.index(
            max(model_results, key=operator.itemgetter(1))
        )
        best_model_params = models[best_model_index][0]
        best_model = models[best_model_index][1]
        return best_model, best_model_params

    def best_model_of_sklearn(self) -> tuple[DecisionTreeClassifier, tuple]:
        """
        Evaluates sklearn's implementation of the DecisionTree
        with defined parameters and returns the best model and parameters

        return: (DecisionTreeClassifier, parameters)
        """
        print("\nCross validation of sklearn DecisionTree...")
        models = [
            ("gini", DecisionTreeClassifier(criterion="gini", random_state=self.seed)),
            (
                "entropy",
                DecisionTreeClassifier(criterion="entropy", random_state=self.seed),
            ),
        ]

        # Evaluate each model
        model_results = []
        kfold = KFold(n_splits=10, random_state=self.seed, shuffle=True)

        for params, model in models:
            validation_scores = []

            for train_index, val_index in kfold.split(self.X_train, self.y_train):
                x_train_fold, x_val_fold = (
                    self.X_train.iloc[train_index],
                    self.X_train.iloc[val_index],
                )

                y_train_fold, y_val_fold = (
                    self.y_train.iloc[train_index],
                    self.y_train.iloc[val_index],
                )

                model.fit(x_train_fold, y_train_fold)
                val_prediction = model.predict(x_val_fold)
                val_score = accuracy_score(y_val_fold, val_prediction)
                validation_scores.append(val_score)

            mean_score = np.mean(validation_scores)
            model_results.append((model, mean_score))
            print(f"{params}: {mean_score}")

        # Test score on best model
        best_model_index = model_results.index(
            max(model_results, key=operator.itemgetter(1))
        )
        best_model_params = models[best_model_index][0]
        best_model = models[best_model_index][1]
        return best_model, best_model_params

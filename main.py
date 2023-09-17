import pandas as pd
import numpy as np
from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import itertools
import operator

seed = 200

data = pd.read_csv("wine_dataset.csv")

X = data.loc[:, :"alcohol"]
y = data["type"]

X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# Creating every possible model for evaluation

impurity_measures = ["entropy", "gini"]
pruning = [True, False]

model_parameters = list(itertools.product(*[impurity_measures, pruning]))


models = [(params, DecisionTree(random_state=seed)) for params in model_parameters]


# Evaluate each model
model_results = []
kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
for params, model in models:
    validation_scores = []
    for train_index, val_index in kfold.split(X_train, y_train):
        x_train_fold, x_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        model.learn(x_train_fold, y_train_fold, params[0], params[1])
        val_prediction = model.predict(x_val_fold)
        val_score = accuracy_score(y_val_fold, val_prediction)
        validation_scores.append(val_score)
    mean_score = np.mean(validation_scores)
    model_results.append((model, mean_score))
    print(f"{params}: {mean_score}")

# Test score on best model
best_model_index = model_results.index(max(model_results, key=operator.itemgetter(1)))
best_model_params = models[best_model_index][0]
best_model = models[best_model_index][1]

best_model.learn(X_train, y_train, best_model_params[0], best_model_params[1])
prediction = best_model.predict(x_test)
score = accuracy_score(y_test, prediction)

print(f"Best model with parameters: {best_model_params}.")
print(f"accuracy: {score}")

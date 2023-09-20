import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Evaluation import Evaluation
from time import time

seed = 15

print(f"Run using seed: {seed}")

# Read dataset and split into training and test data
data = pd.read_csv("wine_dataset.csv")

X = data.loc[:, :"alcohol"]
y = data["type"]

X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# Evaluate the parameters for models
eval = Evaluation(X_train, y_train, random_state=seed)

implementation_model, implementation_params = eval.best_model_of_implementation()
sklearn_model, sklearn_params = eval.best_model_of_sklearn()


# Train and test each model
# and measure training and test time

impl_start = time()
implementation_model.learn(
    X_train, y_train, implementation_params[0], implementation_params[1]
)
prediction = implementation_model.predict(x_test)
impl_end = time()

score = accuracy_score(y_test, prediction)

sk_start = time()
sklearn_model.fit(X_train, y_train)
sklearn_prediction = sklearn_model.predict(x_test)
sk_end = time()

sklearn_score = accuracy_score(y_test, sklearn_prediction)

# Results
print(f"\nBest implemented model with parameters: {implementation_params}.")
print(f"accuracy: {score}")
print(f"Time training and testing: {impl_end-impl_start}")


print(f"\nBest sklearn model with parameters: {sklearn_params}")
print(f"accuracy: {sklearn_score}")
print(f"Time training and testing: {sk_end-sk_start}")

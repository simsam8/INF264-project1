import pandas as pd
from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

seed = 156

data = pd.read_csv("wine_dataset.csv")

X = data.loc[:, :"alcohol"]
y = data["type"]

X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

dt = DecisionTree()

dt.learn(X_train, y_train)

prediction = dt.predict(x_test)

score = accuracy_score(y_test, prediction)

print(score)

# sk_model = DecisionTreeClassifier()
# sk_model.fit(X_train, y_train)
# sk_preds = sk_model.predict(x_test)
# sk_score = accuracy_score(y_test, sk_preds)
#
# print("Implemented tree: ", score)
# print("Sklearn tree: ", sk_score)

# comparison = [(x, y) for x, y in zip(prediction, y_test)]
#
# [print(pair) for pair in comparison]

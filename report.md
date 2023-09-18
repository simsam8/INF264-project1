# Implementing decision trees Report


## Description of data and approach


## Code
The packages used in this project are as follows:

- numpy: matrix operations in implementation
- pandas: initial read of dataset
- sklearn: data splits, accuracy score, and cross validation
- copy(std library): deepcopy for pruning
- collections(std library): counting majority label

The implementation has been written in an object oriented fashion.
The code is divided into four parts, each with their own task:

- DecisionTree.py
- Node.py
- Evaluation.py
- main.py

DecisionTree.py is the class implementing the decision tree.
Node.py is a class representing a node in the decision tree.
Evaluation.py is a class for evaluating and finding the best model
for different implementations.
main.py brings the whole program together and
trains, evaluates and tests models, then outputs the results


### Node.py
The Node class has parameters which can be set,
depending on if it is a leaf node or decision node.
In addition, it has four class methods.

- is_leaf: checks if the node is a leaf node
- convert_to_leaf: converts the node into a leaf node. Used when pruning
- restore_decision_node: converts a leaf node into a decision node. Used when pruning
- display: Displays node information. Used for debugging and displaying tree


### DecisionTree.py
The DecisionTree class implements a decision tree and is initialized with or without a random state.
The class contains two main functions, learn and predict, plus a couple of helper functions.
I'll start with the predict method, which is the shortest of the two main methods.

#### predict

The predict method takes a dataset of unlabeled features.
For each row in the dataset, it calls the recursive helper method _predict. 
_predict traverses the tree, choosing a path depending on the treshold for each node,
until it reaches a leaf node. The value of the leaf node is then stored in an array.

#### learn

The learn method takes four parameters, features, labels, impurity measure, and prune.
Impurity measure and prune are optional, and by default set to "entropy" and False.
Whichever impurity measure is choosen, sets the class' impurity function.
The tree is then build recursively with the helper method _build_tree, and is stored
in the variable root_node.

If pruning is set to True, it splits the training data into 85% training and 15% pruning data
before building the tree.
After the tree is built it prunes the tree. I will come back to pruning at the end of this section.


##### _build_tree

This helper method recursively builds the decision tree.
It's parameters are feature columns, labels, and depth.
There is two stopping criteria:

- there are only one unique label
- the rows of all features are equal

In these cases the method returns a leaf node with the majority label as value.

Then it calculates the best possible split with the method _best_split,
and splits the data on the best feature and its threshold.
It recursively calls itself on each branch and then returns a decision node


##### _best_split

This helper function calculates the best split of the input data.
For every feature column, and every unique value in that feature,
it calculates the information gain.
It keeps track of the best feature column and threshold to split on.


##### _information_gain

This method calculates the information gain by splitting the dataset on 
a feature and threshold.
It uses the impurity measure set in the learn method.


##### _calculate_entropy and _calculate_gini

Calculates the entropy/gini on given labels.
It counts the amount of each label, calculates their probabilities,
and then the entropy/gini using the respective formula.

~~~python
    def _calculate_entropy(self, y):
        probablities = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in probablities if p > 0])

    def _calculate_gini(self, y):
        probabilities = np.bincount(y) / len(y)
        return 1 - np.sum(probabilities**2)
~~~


##### prune

When pruning the decision tree, it first creates a deepcopy of the decision tree.
The pruning itself is implemented using depth first search.

First we create an empty set of visited nodes.
Then we call the the recursive helper function _prune with the set of
visited nodes and the unpruned tree.

The steps of the recursive function are as follows:

1. If the current node is None, do nothing
2. If the current node is not visited, add it to the set 
3. If the current node is a leaf node, convert the previous 
node to a leaf node, and remove it's children. Else explore 
left and right branches.
4. If the accuracy of the pruned tree is worse, restore the nodes and contiune pruning.
Else continue pruning.


### Evaluation.py



## Results

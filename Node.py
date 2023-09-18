class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        data_left=None,
        data_right=None,
        majority_label=None,
        *,
        value=None,
    ) -> None:
        self.data_left: Node = data_left
        self.data_right: Node = data_right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.majority_label = majority_label

    def is_leaf(self) -> bool:
        return self.value is not None

    def convert_to_leaf(self):
        """
        Converts decision node to leaf node:
        """
        self.value = self.majority_label
        self.feature = None
        self.threshold = None
        self.data_left = None
        self.data_right = None

    def restore_decision_node(self, feature, threshold, data_left, data_right):
        """
        Restore node back to decision node
        """
        self.value = None
        self.feature = feature
        self.threshold = threshold
        self.data_right = data_right
        self.data_left = data_left

    def display(self, level) -> None:
        """
        Display node information
        """
        if self.is_leaf():
            print(level * "|", f"Leaf value: {self.value}, Level: {level}")
        else:
            print(
                level * "|",
                f"Feature: {self.feature}, Threshold: {self.threshold}, Level: {level}",
            )

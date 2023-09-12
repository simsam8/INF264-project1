from __future__ import annotations
import pandas as pd


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        data_left: Node = None,
        data_right: Node = None,
        *,
        value=None,
    ) -> None:
        self.data_left: Node = data_left
        self.data_right: Node = data_right
        self.feature = feature
        self.threshold = threshold
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None

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

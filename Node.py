import pandas as pd


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        data_left=None,
        data_right=None,
        gain=None,
        value=None,
    ) -> None:
        self.data_left = data_right
        self.data_right = data_right
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.value = value

    def display_decision(self, level: int) -> None:
        print(f"Level: {level}, Feature: {self.feature}, Threshold: {self.threshold}")

    def display_leaf(self, level: int) -> None:
        print(f"Level: {level}, Value: {self.value}")

import torch.nn as nn


# class ExampleNet(torch.nn):
class ExampleNet():
    """
    This Neural Network does nothing! Woohoo!!!!
    """
    def __init__(self):
        super(ExampleNet, self).__init__()

    def forward(self, x):
        return x
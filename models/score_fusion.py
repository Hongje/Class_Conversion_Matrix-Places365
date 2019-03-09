import torch
import torch.nn as nn
import torch.nn.functional as F

class score_sum(nn.Module):
    def __init__(self, imagenet_model, places_model, contextgating_model):
        super(score_sum, self).__init__()
        self.imagenet_model = imagenet_model
        self.places_model = places_model
        self.contextgating_model = contextgating_model

    def forward(self, x):
        places_score = self.places_model(x)
        image_score = self.imagenet_model(x)
        out = self.contextgating_model(places_score, image_score)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class score_sum_transfer_learning(nn.Module):
    def __init__(self, imagenet_model, places_model, transferlearning_model, contextgating_model):
        super(score_sum_transfer_learning, self).__init__()
        self.imagenet_model = imagenet_model
        self.places_model = places_model
        self.transferlearning_model = transferlearning_model
        self.contextgating_model = contextgating_model

    def forward(self, x):
        places_score = self.places_model(x)
        places_score = self.transferlearning_model(places_score)
        image_score = self.imagenet_model(x)
        out = self.contextgating_model(places_score, image_score)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


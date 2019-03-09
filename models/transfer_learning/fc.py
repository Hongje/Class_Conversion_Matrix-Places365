import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, input_data_length=512, class_num=365):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_data_length, class_num)

    def forward(self, x_places):
        out = self.fc(x_places)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class ReLU_FC(nn.Module):
    def __init__(self, input_data_length=512, class_num=365):
        super(ReLU_FC, self).__init__()
        self.fc = nn.Linear(input_data_length, class_num)

    def forward(self, x_places):
        out = F.relu(x_places)
        out = self.fc(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class feature_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_sum, self).__init__()

        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out = x_places + x_imagenet
        out = self.classifier(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class feature_concat(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_concat, self).__init__()
        self.classifier = nn.Linear(sum(input_data_length), class_num)

    def forward(self, x_places, x_imagenet):
        out = torch.cat((x_places, x_imagenet), dim=-1)
        out = self.classifier(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out





class feature_CCM(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM, self).__init__()

        self.fc1 = nn.Linear(input_data_length[1], input_data_length[1])
        self.fc2 = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = self.fc2(out + x_places)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class feature_CCM_BN(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_BN, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], input_data_length[1], bias=False)
        self.batch_norm = nn.BatchNorm1d(input_data_length[1])

        self.fc2 = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = self.batch_norm(out)
        out = self.fc2(out + x_places)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out




class feature_CCM_ReLU(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_ReLU, self).__init__()

        self.fc1 = nn.Linear(input_data_length[1], input_data_length[0])
        self.fc2 = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = F.relu(out)
        out = self.fc2(out + x_places)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class feature_CCM_BN_ReLU(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_BN_ReLU, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], input_data_length[0], bias=False)
        self.batch_norm = nn.BatchNorm1d(input_data_length[1])

        self.fc2 = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.fc2(out + x_places)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class feature_CCM_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)

        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.softmax(weighted, dim=-1)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))

        out = self.classifier(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class feature_CCM_BN_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_BN_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0], bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = self.batch_norm2(out2)
        
        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.softmax(weighted, dim=-1)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))
        
        out = self.classifier(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

class feature_CCM_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)

        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.sigmoid(weighted)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))

        out = self.classifier(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class feature_CCM_BN_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_BN_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0], bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = self.batch_norm2(out2)
        
        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.sigmoid(weighted)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))
        
        out = self.classifier(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class feature_CCM_ReLU_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_ReLU_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = F.relu(out2)

        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.softmax(weighted, dim=-1)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))

        out = self.classifier(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class feature_CCM_BN_ReLU_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_BN_ReLU_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0], bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = self.batch_norm2(out2)
        out2 = F.relu(out2)
        
        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.softmax(weighted, dim=-1)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))
        
        out = self.classifier(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

class feature_CCM_ReLU_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_ReLU_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = F.relu(out2)

        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.sigmoid(weighted)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))

        out = self.classifier(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class feature_CCM_BN_ReLU_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[2048, 2048], class_num=365):
        super(feature_CCM_BN_ReLU_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], input_data_length[0], bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)


        self.classifier = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = self.batch_norm2(out2)
        out2 = F.relu(out2)
        
        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.sigmoid(weighted)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))
        
        out = self.classifier(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


class FC1_only_imagenet(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_only_imagenet, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_only_imagenet(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_only_imagenet, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm = nn.BatchNorm1d(input_data_length[0])

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = self.batch_norm(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class FC1_only_place(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_only_place, self).__init__()
        self.fc1 = nn.Linear(input_data_length[0], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_places)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_only_place(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_only_place, self).__init__()
        self.fc1 = nn.Linear(input_data_length[0], class_num, bias=False)
        self.batch_norm = nn.BatchNorm1d(input_data_length[0])

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_places)
        out = self.batch_norm(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class FC1_imagenet_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenet_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = out + x_places

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenet_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenet_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm = nn.BatchNorm1d(input_data_length[0])

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = self.batch_norm(out)
        out = out + x_places

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

class FC1_imagenetReLU_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenetReLU_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num)

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = F.relu(out)
        out = out + x_places

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenetReLU_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenetReLU_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm = nn.BatchNorm1d(input_data_length[0])

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = out + x_places

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

class FC1_imagenetReLU_bias_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenetReLU_bias_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num)
        self.bias = nn.Parameter(torch.Tensor(class_num))

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = F.relu(out)
        out = out + self.bias
        out = out + x_places

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenetReLU_bias_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenetReLU_bias_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm = nn.BatchNorm1d(input_data_length[0])
        self.bias = nn.Parameter(torch.Tensor(class_num))

    def forward(self, x_places, x_imagenet):
        out = self.fc1(x_imagenet)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = out + self.bias
        out = out + x_places

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class FC1_imagenet_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenet_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num)

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenet_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenet_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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
        
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

class FC1_imagenet_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenet_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num)

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenet_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenet_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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
        
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class FC1_imagenetReLU_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenetReLU_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num)

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenetReLU_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenetReLU_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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
        
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out



class FC1_imagenetReLU_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenetReLU_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num)

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenetReLU_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenetReLU_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

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
        
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class FC1_imagenetReLU_bias_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenetReLU_bias_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num)
        self.bias = nn.Parameter(torch.Tensor(class_num))

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = F.relu(out2)
        out2 = out2 + self.bias

        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.softmax(weighted, dim=-1)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenetReLU_bias_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenetReLU_bias_weighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])
        self.bias = nn.Parameter(torch.Tensor(class_num))

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = self.batch_norm2(out2)
        out2 = F.relu(out2)
        out2 = out2 + self.bias
        
        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.softmax(weighted, dim=-1)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))
        
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class FC1_imagenetReLU_bias_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_imagenetReLU_bias_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num)
        self.bias = nn.Parameter(torch.Tensor(class_num))

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = F.relu(out2)
        out2 = out2 + self.bias

        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.sigmoid(weighted)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC1_BN_imagenetReLU_bias_Sigmoidweighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC1_BN_imagenetReLU_bias_Sigmoidweighted_sum, self).__init__()
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])
        self.bias = nn.Parameter(torch.Tensor(class_num))

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

    def forward(self, x_places, x_imagenet):
        out2 = self.fc2(x_imagenet)
        out2 = self.batch_norm2(out2)
        out2 = F.relu(out2)
        out2 = out2 + self.bias
        
        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.sigmoid(weighted)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (x_places * weighted[:,0].unsqueeze(-1).expand_as(x_places))
        
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out









class FC_imagenet_FC_place_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC_imagenet_FC_place_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[0], class_num)
        self.fc2 = nn.Linear(input_data_length[1], class_num)

    def forward(self, x_places, x_imagenet):
        out1 = self.fc1(x_places)
        out2 = self.fc2(x_imagenet)
        out = out1 + out2

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC_BN_imagenet_FC_BN_place_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC_BN_imagenet_FC_BN_place_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[0], class_num, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(input_data_length[0])
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

    def forward(self, x_places, x_imagenet):
        out1 = self.fc1(x_places)
        out1 = self.batch_norm1(out1)
        out2 = self.fc2(x_imagenet)
        out2 = self.batch_norm2(out2)
        out = out1 + out2

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class FC_imagenet_FC_place_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC_imagenet_FC_place_weighted_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[0], class_num)
        self.fc2 = nn.Linear(input_data_length[1], class_num)

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

    def forward(self, x_places, x_imagenet):
        out1 = self.fc1(x_places)
        out2 = self.fc2(x_imagenet)

        weighted_pl = self.weighted_fc11(x_places)
        weighted_pl = F.relu(weighted_pl)
        weighted_pl = self.weighted_fc12(weighted_pl)

        weighted_im = self.weighted_fc21(x_imagenet)
        weighted_im = F.relu(weighted_im)
        weighted_im = self.weighted_fc22(weighted_im)

        weighted = torch.cat((weighted_pl, weighted_im), dim=-1)
        weighted = F.softmax(weighted, dim=-1)

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (out1 * weighted[:,0].unsqueeze(-1).expand_as(x_places))

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


        
class FC_BN_imagenet_FC_BN_place_weighted_sum(nn.Module):
    def __init__(self, input_data_length=[365,1000], class_num=365):
        super(FC_BN_imagenet_FC_BN_place_weighted_sum, self).__init__()
        self.fc1 = nn.Linear(input_data_length[0], class_num, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(input_data_length[0])
        self.fc2 = nn.Linear(input_data_length[1], class_num, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(input_data_length[0])

        reduction_ratio = 8
        weighted_places365_feature_num = input_data_length[0] // reduction_ratio
        weighted_imagenet_feature_num = input_data_length[1] // reduction_ratio
        self.weighted_fc11 = nn.Linear(input_data_length[0], weighted_places365_feature_num)
        self.weighted_fc12 = nn.Linear(weighted_places365_feature_num, 1)
        self.weighted_fc21 = nn.Linear(input_data_length[1], weighted_imagenet_feature_num)
        self.weighted_fc22 = nn.Linear(weighted_imagenet_feature_num, 1)

    def forward(self, x_places, x_imagenet):
        out1 = self.fc1(x_places)
        out1 = self.batch_norm1(out1)
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

        out = (out2 * weighted[:,1].unsqueeze(-1).expand_as(x_places)) + (out1 * weighted[:,0].unsqueeze(-1).expand_as(x_places))
        
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

# class FC1_bilinear(nn.Module):
#     def __init__(self, input_data_length=[365,1000], class_num=365):
#         super(FC1_bilinear, self).__init__()
#         self.fc1 = nn.Bilinear(input_data_length[0], input_data_length[1], class_num)

#     def forward(self, x_places, x_imagenet):
#         out = self.fc1(x_places, x_imagenet)

#         # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
#         return out

class FC1_bilinear(nn.Module):
    def __init__(self, input_data_length=[2208,2208], class_num=365, feature_num=128):
        super(FC1_bilinear, self).__init__()
        self.fc1 = nn.Linear(input_data_length[0], feature_num)
        self.fc2 = nn.Linear(input_data_length[1], feature_num)

        self.fusion = nn.Bilinear(feature_num, feature_num, class_num)

    def forward(self, x_places, x_imagenet):
        x_places = self.fc1(x_places)
        x_imagenet = self.fc2(x_imagenet)

        out = self.fusion(x_places, x_imagenet)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

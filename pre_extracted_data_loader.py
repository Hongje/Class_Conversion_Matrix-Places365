# coding: utf-8

import torch
from torch.utils.data import Dataset

import torchvision

import numpy as np
import os
import sys



class pre_extracted_dataset(Dataset):
    train_list = [
        'places365/train.npz',
        'imagenet/train.npz',
    ]
    test_list = [
        'places365/validate.npz',
        'imagenet/validate.npz',
    ]
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set


        if self.train:
            f = self.train_list[0]
            file_path = os.path.join(self.root, f)
            fo = np.load(file_path)
            self.train_data_places = fo['data']
            self.train_labels = fo['label']
            fo.close()

            f = self.train_list[1]
            file_path = os.path.join(self.root, f)
            fo = np.load(file_path)
            self.train_data_imagenet = fo['data']
            fo.close()


        else:
            f = self.test_list[0]
            file_path = os.path.join(self.root, f)
            fo = np.load(file_path)
            self.test_data_places = fo['data']
            self.test_labels = fo['label']
            fo.close()

            f = self.test_list[1]
            file_path = os.path.join(self.root, f)
            fo = np.load(file_path)
            self.test_data_imagenet = fo['data']
            fo.close()


    def __getitem__(self, index):

        if self.train:
            data_places = self.train_data_places[index]
            data_imagenet = self.train_data_imagenet[index]
            target = self.train_labels[index]
        else:
            data_places = self.test_data_places[index]
            data_imagenet = self.test_data_imagenet[index]
            target = self.test_labels[index]

        if self.transform is not None:
            data_places = self.transform(data_places)
            data_places = self.transform(data_imagenet)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return data_places, data_imagenet, target


    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)





class pre_extracted_places_feature(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set


        if self.train:
            file_path = self.root
            fo = np.load(file_path)
            self.train_data_places = fo['data']
            self.train_labels = fo['label']
            fo.close()

        else:
            file_path = self.root
            fo = np.load(file_path)
            self.test_data_places = fo['data']
            self.test_labels = fo['label']
            fo.close()


    def __getitem__(self, index):

        if self.train:
            data_places = self.train_data_places[index]
            target = self.train_labels[index]
        else:
            data_places = self.test_data_places[index]
            target = self.test_labels[index]

        if self.transform is not None:
            data_places = self.transform(data_places)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return data_places, target


    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)


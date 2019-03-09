# coding: utf-8

import torch
from torch.utils.data import Dataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class places_dataset(Dataset):
    """ Coview 2018 frame Dataset
    This code was created with reference to pytorch's IMAGENET DATASET code
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    """
    
    def __init__(self, root, filelist_folder, transform=None, target_transform=None,
                 train=True, loader=default_loader):
    # def __init__(self, root, train=True, rgb_audio_concat=True, transform=None,
    #              target_transform=None, max_data_temporal_length=300,
    #              only_scene=False, only_action=False):

        filelists = [f for f in os.listdir(filelist_folder) if os.path.isfile(os.path.join(filelist_folder, f))]

        if train:
            if 'places365_train_standard.txt' in filelists:
                filelist = os.path.join(filelist_folder, 'places365_train_standard.txt')
            elif 'places365_train_challenge.txt' in filelists:
                filelist = os.path.join(filelist_folder, 'places365_train_challenge.txt')
            else:
                raise(RuntimeError('Error: CANNOT find file list\n(places365_train_standard.txt) or\n(places365_train_challenge.txt) is needed'))

        else:
            if 'places365_val.txt' in filelists:
                filelist = os.path.join(filelist_folder, 'places365_val.txt')
            else:
                raise(RuntimeError('Error: CANNOT find file list\n(places365_val.txt) is needed'))


        if 'categories_places365.txt' in filelists:
            class_name_file = os.path.join(filelist_folder, 'categories_places365.txt')
            classes_load = []
            with open(class_name_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    field = line.strip().split(' ')
                    if len(field) == 2:
                        classes_load.append((field[0], int(field[1])))
                    else:
                        raise(RuntimeError('Error: The class lists line should be [str(class name), int(class_idx)]'))
            classes = [''] * len(classes_load)
            for i in range(len(classes_load)):
                classes[classes_load[i][1]] = classes_load[i][0]
        
        else:
            classes = None

        samples = []
        with open(filelist, 'r') as f:
            lines = f.readlines()
            for line in lines:
                field = line.strip().split(' ')
                if len(field) == 2:
                    samples_path = field[0]
                    if samples_path[0] == '/':
                        samples_path = samples_path[1:]
                    samples_path = os.path.join(root, samples_path)
                    class_idx = int(field[1])
                    samples.append((samples_path, class_idx))
                else:
                    raise(RuntimeError('Error: The file lists line should be [str(image_path), int(class_idx)]'))
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in filists of: " + filelist))

        self.root = root
        self.loader = loader

        self.samples = samples
        self.imgs = self.samples
        self.classes = classes
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):       
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)


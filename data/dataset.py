'''
@env_version: python3.5.2
@Author: 雷国栋
@LastEditors: 雷国栋
@Date: 2020-02-10 17:10:43
@LastEditTime: 2020-02-22 15:42:44
'''

import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import random
from PIL import Image
from torchvision import transforms as tfs
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.getcwd())
from baseset import configs

import csv

random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
torch.cuda.manual_seed_all(configs.seed)


def general_csv(root):
    words = os.listdir(root)
    if not os.path.exists('comparison_table.csv'):
        os.system(r"touch {}".format('comparison_table.csv'))
        with open('comparison_table.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['sign', 'charact'])
        j = 0
        for p in words:
            with open('comparison_table.csv', 'a+', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow([j, p])
            j += 1


def general_tvt(root):
    words = os.listdir(root)
    sign = True
    if not os.path.exists('./data_split_list/trains.txt'):
        # os.remove('./data_split_list/trains.txt')
        os.system(r"touch {}".format('./data_split_list/trains.txt'))
        sign = False
    if not os.path.exists('./data_split_list/tests.txt'):
        # os.remove('./data_split_list/tests.txt')
        os.system(r"touch {}".format('./data_split_list/tests.txt'))
        sign = False
    if not os.path.exists('./data_split_list/valids.txt'):
        # os.remove('./data_split_list/valids.txt')
        os.system(r"touch {}".format('./data_split_list/valids.txt'))
        sign = False

    if sign:
        return
    label = 0
    for p in words:
        files = root + p
        imgs = []
        labels = []
        for i in os.listdir(files):
            img = os.path.join(files, i)
            imgs.append(img)
            labels.append(label)
        train, valid, train_label, valid_label = train_test_split(
            imgs,
            labels,
            shuffle=True,
            test_size=0.05,
            random_state=random.randint(0, 20))
        test, valid, test_label, valid_label = train_test_split(
            valid,
            valid_label,
            shuffle=True,
            test_size=0.5,
            random_state=random.randint(0, 2))

        with open("./data_split_list/trains.txt", 'a+') as f:
            for k, l in zip(train, train_label):
                f.write(k + ',' + str(l) + '\n')

        with open("./data_split_list/tests.txt", 'a+') as f:
            for k, l in zip(test, test_label):
                f.write(k + ',' + str(l) + '\n')

        with open("./data_split_list/valids.txt", 'a+') as f:
            for k, l in zip(valid, valid_label):
                f.write(k + ',' + str(l) + '\n')
        label += 1


def make_dataset(file_side):
    imgs = []
    f = open(file_side, 'r', encoding='utf-8')
    all_lines = f.readlines()
    for line in all_lines:
        img, label = line.split(',')
        imgs.append((img, int(label)))
    f.close()
    return imgs


class CharacterDataset(Dataset):
    def __init__(self,
                 root,
                 arg=False,
                 mean=None,
                 std=None,
                 brightness=0.1,
                 angle=15,
                 up_biase=0.04,
                 left_biase=0.04):
        self.data = make_dataset(root)
        self.arg = arg
        self.mean = mean
        self.std = std
        self.brightness = brightness
        self.angle = angle
        self.up_biase = up_biase
        self.left_biase = left_biase

    def update_arg(self, arg):
        self.arg = arg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        tfs_parameter = []
        if self.arg:
            angle = random.uniform(-self.angle, self.angle)
            bright = random.uniform(-self.brightness, self.brightness)
            tfs_parameter.append(tfs.RandomRotation(angle))
            color_jitter = tfs.ColorJitter(bright)
            tfs_parameter.append(color_jitter)
        tfs_parameter.append(tfs.ToTensor())
        if not self.mean is None and not self.std is None:
            normal = tfs.Normalize(mean=self.mean, std=self.std)
            tfs_parameter.append(normal)
        transform = tfs.Compose(tfs_parameter)
        img = transform(img)
        img = tfs.ToPILImage()(img)
        img = img.resize(configs.img_size, Image.ANTIALIAS)
        img = tfs.ToTensor()(img)
        label = torch.FloatTensor([label])
        return img, label


class Get_data():
    def __init__(self, train, valid, test):
        general_csv(configs.train_side)
        general_tvt(configs.train_side)
        self.train_dataset = CharacterDataset(train)
        self.valid_dataset = CharacterDataset(valid)
        self.test_dataset = CharacterDataset(test)

    def get_train_data(self):
        train_dataloader = DataLoader(self.train_dataset, configs.batch_size,
                                      configs.shuffle)
        return train_dataloader

    def get_valid_data(self):
        valid_dataloader = DataLoader(self.valid_dataset, configs.batch_size,
                                      configs.shuffle)
        return valid_dataloader

    def get_test_data(self):
        test_dataloader = DataLoader(self.test_dataset, configs.batch_size,
                                     configs.shuffle)
        return test_dataloader


if __name__ == "__main__":
    general_tvt(configs.train_side)
    # gd = Get_data("./data_split_list/trains.txt",
    #               "./data_split_list/valids.txt",
    #               "./data_split_list/tests.txt")
    # print(len(gd.train_dataset))
    # print(len(gd.valid_dataset))
    # print(len(gd.test_dataset))
    # print(gd.get_test_data())

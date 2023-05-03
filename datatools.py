import torch
import os
import csv

from torch.utils import data

class CKextendDataset():
    def __init__(self, path, csvfile):
        super().__init__()
        
        self.path = path
        self.csvfile = csvfile
        fullpath = os.path.join(self.path, self.csvfile)

        train_y = []
        train_x = []
        
        valid_y = []
        valid_x = []
        
        test_y = []
        test_x = []
        
        if os.path.exists(fullpath):
            with open(fullpath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, line in enumerate(reader):
                    if i == 0:
                        continue
                    if 'Training' in line[2]:
                        train_y.append(int(line[0]))
                        train_x.append([int(i) for i in str.split(line[1], ' ')])
                    elif 'PublicTest' in line[2]:
                        valid_y.append(int(line[0]))
                        valid_x.append([int(i) for i in str.split(line[1], ' ')])
                    elif 'PrivateTest' in line[2]:
                        test_y.append(int(line[0]))
                        test_x.append([int(i) for i in str.split(line[1], ' ')])
        
        self.train_y = torch.Tensor(train_y)
        self.train_x = torch.Tensor(train_x).view(-1, 48, 48).contiguous()
        self.valid_y = torch.Tensor(valid_y)
        self.valid_x = torch.Tensor(valid_x).view(-1, 48, 48).contiguous()
        self.test_y = torch.Tensor(test_y)
        self.test_x = torch.Tensor(test_x).view(-1, 48, 48).contiguous()
        
class TrainDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset.train_y)
    def __getitem__(self, idx):
        return self.dataset.train_x[idx], self.dataset.train_y[idx]

class ValidDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset.valid_y)
    def __getitem__(self, idx):
        return self.dataset.valid_x[idx], self.dataset.valid_y[idx]

class TestDataset(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset.test_y)
    def __getitem__(self, idx):
        return self.dataset.test_x[idx], self.dataset.test_y[idx]
        
dataset = CKextendDataset('./dataset','ckextended.csv')
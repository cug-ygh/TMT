import os.path

from torch.utils.data.dataset import Dataset
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import pickle
from torch.utils.data.dataloader import default_collate 


def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        batch_tmp = {
            'text': _batch['text'], 
            'audio': _batch['audio'],
            'vision': _batch['vision'],
            'labels': _batch['labels'],
            'index': _batch['index'],
        } 
        new_batch.append(batch_tmp)
        ids.append(_batch['id'])
    ids = np.stack(ids, axis=0)
    return default_collate(new_batch), ids

class MOSI(Dataset):
    def __init__(self, dataPath, mode):
        self.dataPath = dataPath
        self.mode = mode
        with open(self.dataPath, 'rb') as f:
            data = pickle.load(f)
        self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.ids = data[self.mode]['id']
        self.label = data[self.mode]['labels'].astype(np.float32)


    def __getitem__(self, index):
        label = self.label[index].reshape(-1)
        # if label >= 2.5:
        #     label = 6
        # elif label >=1.5:
        #     label = 5
        # elif label >= 0.5:
        #     label = 4
        # elif label >= -0.5:
        #     label = 3
        # elif label >= -1.5:
        #     label = 2
        # elif label >= -2.5:
        #     label = 1
        # elif label >= -3:
        #     label = 0

        if label < 0:
            label = 0
        elif label == 0:
            label = 1
        elif label >0:
            label = 2

        sample = {
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': torch.Tensor([label]),
            'index': index,
            'id': self.ids[index]
        } 

        return sample

    def __len__(self):
        return self.ids.shape[0]


def load_dataset(dataPath, batchsize):
    train_set = MOSI(dataPath=dataPath, mode='train')
    tqdm_train_total = math.ceil(train_set.__len__() / float(batchsize))
    print(tqdm_train_total)
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True, drop_last=False, num_workers=0, collate_fn=id_collate)

    valid_set = MOSI(dataPath=dataPath, mode='valid')
    tqdm_valid_total = math.ceil(valid_set.__len__() / float(batchsize))
    print(tqdm_valid_total)
    valid_loader = DataLoader(valid_set, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=0, collate_fn=id_collate)

    test_set = MOSI(dataPath=dataPath, mode='test')
    tqdm_test_total = math.ceil(test_set.__len__() / float(batchsize))
    print(tqdm_test_total)
    test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=0, collate_fn=id_collate)

    return train_loader, tqdm_train_total, valid_loader, tqdm_valid_total, test_loader, tqdm_test_total


if __name__ == '__main__':

    train_set = MOSI(dataPath=r"D:\Users\Home\Downloads\seq_length_50\mosi_data.pkl", mode='test')
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=False, num_workers=0, collate_fn=id_collate)

    for iter, sample in enumerate(train_loader):
        print(sample[0].keys())
        pass

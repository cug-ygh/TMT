import os
import datetime
import shutil
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import random

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()
    return n_correct_elements / batch_size

class AverageMeter(object):

    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count

def save_model(save_path, epoch, model, optimizer, ACC2, ACC3, Has0_F1_score, depth):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'save_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.pth'.format(epoch, ACC2, Has0_F1_score, ACC3, depth))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)

def save_model_reg(save_path, epoch, model, optimizer, ACC2, ACC3, ACC5, F1, MAE, CORR, depth):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'save_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_d{}.pth'.format(epoch, ACC2, ACC3, ACC5, F1, MAE, CORR, depth))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)


def save_model_self_supervision(save_path, epoch, model, optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'save_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)


def normlize(x):

    fenmu = (np.max(x) - np.min(x))
    if fenmu == 0:
        return np.zeros_like(x)
    else:
        a = (x - np.min(x)) / fenmu
        return a



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pretrained_models(model, opt):
    if (opt.visual_pretrain_models is not None) and (opt.train_stage == 2):
        checkpoint = torch.load(opt.visual_pretrain_models)['state_dict']
        checkpoint.pop("cls_head.0.weight")
        checkpoint.pop("cls_head.0.bias")
        checkpoint.pop("cls_head.1.weight")
        checkpoint.pop("cls_head.1.bias")
        model.load_state_dict(checkpoint, strict=False)
        print("load ", opt.visual_pretrain_models)
    else:
        print("no visual_pretrain_models")

    if (opt.audio_pretrain_models is not None) and (opt.train_stage == 2):
        checkpoint = torch.load(opt.audio_pretrain_models)['state_dict']
        checkpoint.pop("cls_head.0.weight")
        checkpoint.pop("cls_head.0.bias")
        checkpoint.pop("cls_head.1.weight")
        checkpoint.pop("cls_head.1.bias")
        # checkpoint.pop('fc.weight')
        # checkpoint.pop('fc.bias')
        model.load_state_dict(checkpoint, strict=False)
        print("load ", opt.audio_pretrain_models)
    else:
        print("no audio_pretrain_models")

    if (opt.pretrain_models is not None) and (opt.train_stage == 3):
        checkpoint = torch.load(opt.pretrain_models)['state_dict']
        model.load_state_dict(checkpoint)
        print("load ", opt.pretrain_models)
    else:
        print("no pretrain_models")


    if (opt.visual_pretrain_models is not None) and (opt.train_stage == 4):
        checkpoint = torch.load(opt.visual_pretrain_models)['state_dict']
        checkpoint.pop("cls_head.0.weight")
        checkpoint.pop("cls_head.0.bias")
        checkpoint.pop("cls_head.1.weight")
        checkpoint.pop("cls_head.1.bias")
        model.load_state_dict(checkpoint, strict=False)
        print("load ", opt.visual_pretrain_models)
    else:
        print("no visual_pretrain_models")

    if (opt.audio_pretrain_models is not None) and (opt.train_stage == 4):
        checkpoint = torch.load(opt.audio_pretrain_models)['state_dict']
        checkpoint.pop("cls_head.0.weight")
        checkpoint.pop("cls_head.0.bias")
        checkpoint.pop("cls_head.1.weight")
        checkpoint.pop("cls_head.1.bias")
        # checkpoint.pop('fc.weight')
        # checkpoint.pop('fc.bias')
        model.load_state_dict(checkpoint, strict=False)
        print("load ", opt.audio_pretrain_models)
    else:
        print("no audio_pretrain_models")


    return model


def mixup_data(img, audio, text, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = img.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_img = lam * img + (1 - lam) * img[index, :]
    mixed_audio = lam * audio + (1 - lam) * audio[index, :]
    mixed_text = lam * text + (1 - lam) * text[index, :]
    y_a, y_b = y, y[index]

    return mixed_img, mixed_audio, mixed_text, y_a, y_b, lam

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
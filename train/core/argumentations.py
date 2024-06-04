import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
from imgaug import augmenters as iaa

def rotate(video):
    v = random.uniform(-15,15)
    for i in range(len(video)):
        video[i] = video[i].rotate(v)
    return video

def Brightness(video):
    v = random.uniform(0.3,2)
    for i in range(len(video)):
        video[i] = PIL.ImageEnhance.Brightness(video[i]).enhance(v)
    return video

def flip(video):
    for i in range(len(video)):
        video[i] = PIL.ImageOps.mirror(video[i])
    return video

def GaussianNoise(video):
    v = random.uniform(0,8)
    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=v)
    ])
    for i in range(len(video)):
        video[i] = np.array(video[i])
        video[i] = np.expand_dims(video[i], axis=0)
        video[i] = seq(images=video[i]).squeeze()
        video[i] = Image.fromarray(video[i])
    return video


def Crop(video):
    v = random.randint(0, 16)
    seq = iaa.Sequential([
        iaa.Crop(px=v)
    ])
    for i in range(len(video)):
        video[i] = np.array(video[i])
        video[i] = np.expand_dims(video[i], axis=0)
        video[i] = seq(images=video[i]).squeeze()
        video[i] = Image.fromarray(video[i])
    return video


def EnhanceColor(video):
    v = random.uniform(0, 3)
    for i in range(len(video)):
        video[i] = PIL.ImageEnhance.Color(video[i]).enhance(v)
    return video


def EnhanceContrast(video):
    v = random.uniform(0.5, 3)
    for i in range(len(video)):
        video[i] = PIL.ImageEnhance.Contrast(video[i]).enhance(v)
    return video

def EnhanceSharpness(video):
    v = random.uniform(-5, 5)
    for i in range(len(video)):
        video[i] = PIL.ImageEnhance.Sharpness(video[i]).enhance(-5)
    return video


def augment_list(video):
    flag_list = [random.randint(0, 1) for i in range(7)]
    video = Crop(video)
    if flag_list[0] == 1:
        video = rotate(video)
    if flag_list[1] == 1:
        video = Brightness(video)
    if flag_list[2] == 1:
        video = flip(video)
    if flag_list[3] == 1:
        video = EnhanceColor(video)
    if flag_list[4] == 1:
        video = EnhanceContrast(video)
    if flag_list[5] == 1:
        video = EnhanceSharpness(video)
    if flag_list[6] == 1:
        video = GaussianNoise(video)

    return video

if __name__ =='__main__':
    img = Image.open('../001.jpg')
    # img.show(img)
    b = [img, img]
    # print(len(b))
    img_aug = GaussianNoise(b)
    img_aug[0].save('../1.jpg')
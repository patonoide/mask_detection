import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random
from PIL import Image
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

import torch
import torchvision

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os



img_base_path = "./dataset/images"
# Files
images = os.listdir(img_base_path)
print("Number of Images: ", len(images))

# This loader will use the underlying loader plus crop the image based on the annotation
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

annotation_folders = os.listdir('./dataset/annotations')
def ImageLoader(path):
    img = datasets.folder.default_loader(path) # default loader
    # Get bounding box
    breed_folder = [x for x in annotation_folders if path.split('/')[-1].split('_')[0] in x][0]
    annotation_path = os.path.join('./dataset/annotations', breed_folder, path.split('/')[-1][:-4])

    tree = ET.parse(annotation_path)
    root = tree.getroot()
    objects = root.findall('object')
    for obj in objects:
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
    bbox = (xmin, ymin, xmax, ymax)

    # return cropped image
    img = img.crop(bbox)
    img = img.resize((64, 64), Image.ANTIALIAS)
    return img



# Data Pre-procesing and Augmentation (Experiment on your own)
random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]

transform = transforms.Compose([
                                transforms.CenterCrop(64),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply(random_transforms, p=0.3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# The dataset (example)
dataset = torchvision.datasets.ImageFolder(
    './dataset/images',
    loader=ImageLoader, # THE CUSTOM LOADER
    transform=transform
)

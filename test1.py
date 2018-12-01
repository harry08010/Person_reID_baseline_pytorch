# -*- coding: utf-8 -*-
import argparse
import os
import scipy.io
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
from model import ft_net, ft_net_dense, PCB, PCB_test
"""
Created on Fri Nov 30 15:07:02 2018

@author: huaya
"""

class Dataset(Dataset):
    def __init__(self, path, transform):
        self.dir = path
        self.image = [f for f in os.listdir(self.dir) if f.endswith('png')]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        name = self.image[idx]

        img = Image.open(os.path.join(self.dir, name))
        img = self.transform(img)

        return {'name': name.replace('.png', ''), 'img': img}

def extractor(model, dataloader):
    def fliplr(img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    test_names = []
    test_features = torch.FloatTensor()

    for batch, sample in enumerate(dataloader):
        names, images = sample['name'], sample['img']

        ff = model(Variable(images.cuda(), volatile=True)).data.cpu()
        ff = ff + model(Variable(fliplr(images).cuda(), volatile=True)).data.cpu()
        ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))

        test_names = test_names + names
        test_features = torch.cat((test_features, ff), 0)

    return test_names, test_features

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model','ft_ResNet50','net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network


# Load Collected data Trained model
print('-------test-----------')
model_structure = ft_net(751)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer

model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()


# Change to test mode
model = model.eval()
model = model.cuda()

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#image_datasets = {x: datasets.ImageFolder( os.path.join('/home/zzd/Market/pytorch',x) ,data_transforms) for x in ['gallery','query']}
image_datasets ={x: Dataset(path = os.path.join('../Market-1501-v15.09.15/valSet',x),transform=data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=False, num_workers=0) for x in ['gallery','query']}

for dataset in ['valid']:
    for subset in ['query', 'gallery']:
        test_names, test_features = extractor(model, dataloaders[subset])
        results = {'names': test_names, 'features': test_features.numpy()}
        scipy.io.savemat(os.path.join('../log', 'feature_%s_%s.mat' % (dataset, subset)), results)

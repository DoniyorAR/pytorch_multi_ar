# -*- coding: utf-8 -*-

import torchvision.transforms.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
# from utils import train_model
import time
import argparse
import copy
# import numpy as np
import platform, psutil
# import pandas as pd
# import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
# from utils import train_model
import gc
gc.collect()
torch.cuda.empty_cache()

##############################################################################
#
#   COMPUTE MEANS AND STDS WHEN TRAINING ON A NEW DATASET
#

means = [0.6039, 0.5215, 0.4241]
stds =  [0.1572, 0.1731, 0.2067]
data_dir = 'data'
full_path = "/home/tako/dataset/data"
BATCH_SIZE = 48
num_epochs = 100

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
args = parser.parse_args()
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = (600, 600)
# class SquarePad:
#     def __call__(self, image):
#         max_wh = max(image.size)
#         p_left, p_top = [(max_wh - s) // 2 for s in image.size]
#         p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
#         padding = (p_left, p_top, p_right, p_bottom)
#         return F.pad(image, padding, 0, 'constant')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomVerticalFlip(),
        #SquarePad(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.Resize(image_size),
        #transforms.RandomInvert(),
        #transforms.RandomAdjustSharpness(sharpness_factor=2),
        #transforms.Grayscale(num_output_channels=3), #  using grayscale triggers error of
        # output with shape [1, 600, 600] doesn't match the broadcast shape
        #transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=means,std=stds)
                                ]),
    'val': transforms.Compose([
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        #SquarePad(),
        #transforms.Grayscale(num_output_channels=3),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=means,std=stds)
                            ]),
    }



path = {x: os.path.join(os.path.dirname(os.path.abspath(full_path)), data_dir,x)
            for x in ['train', 'val']}
##############################################################################
#   use shuffle

image_datasets = {x: datasets.ImageFolder(path[x], data_transforms[x])
                                          for x in ['train', 'val']}
dataloaders = { 'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=0),
                'val' : torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=0),
                                             }

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

print("dataset_sizes :", dataset_sizes)
inputs, classes = next(iter(dataloaders['train']))
# out = torchvision.utils.make_grid(inputs)




def train_model(model, criterion, optimizer, dataloaders, device, dataset_sizes, scheduler, num_epochs):
    since = time.time()
    time_elapsed = time.time() - since
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # d = datetime.date.today()
    print('Session START :', time.strftime('%Y-%m-%d %Z %H:%M:%S', time.localtime(time.time())))
    print('===============================================================')
    #print(d.isoformat())
    def printOsInfo():

        print('GPU                  :\t', torch.cuda.device_count()) 
        print('OS                   :\t', platform.system())
 
    if __name__ == '__main__':
        printOsInfo()

    def printSystemInfor():
        print('Process information  :\t', platform.processor())
        print('Process Architecture :\t', platform.machine())
        print('RAM Size             :\t',str(round(psutil.virtual_memory().total / (1024.0 **3)))+"(GB)")
        print('===============================================================')
          
    if __name__ == '__main__':
        printSystemInfor()  

    tb = SummaryWriter()
    # ster - rcnn
    # github
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0 ; running_corrects = 0
           
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            time_elapsed = time.time() - since
            print('{} Loss: {:.4f} Acc: {:.4f} Training complete in {:.0f}. {:.0f}.s'.format(
                  phase, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60 ))
            #time_elapsed = time.time() - since    
            #print('Training complete ini {:.0f}. {:.0f}.s'.format(time_elapsed // 60, time_elapsed % 60))
            
            if phase == 'train':
                tb.add_scalar("train_loss", epoch_loss, epoch)
                tb.add_scalar("train_accu", epoch_acc, epoch)
            else:
                tb.add_scalar("val_loss", epoch_loss, epoch)  
                tb.add_scalar("val_acc", epoch_acc, epoch)  

              
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                

        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Session END')
  
    model.load_state_dict(best_model_wts)
    return model


##############################################################################
#
#       For the built-in resnet18
#
#
# model_ft = models.resnet18(pretrained=False)
# num_feats = model_ft.fc.in_features  # num_input_features of the built-in ResNet50
# model_ft.fc = nn.Linear(num_feats, 5)# replace the num_feats by num_feat of our dataset
                                       # 5 : that is, Q1, Q2, Q3, Q4, Q5

##############################################################################
#
#                   Training the model for quantity of soup
#   1. there are 3 soup items in the image dataset
#   2. Use the built-in Mobilenet_v2 and the following hyperparameter setup for soup 
#   3. question: what feature of rgb image is required to collect for estimating weight of food
#
#
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=15)
print(model.classifier)



model = model.to(device)

criterion = nn.CrossEntropyLoss() 

###############################################################################
#   Add L1/L2 regularization in PyTorch
#   ====> weight_decay=1e-2: the reason why 1e-2 is chosen is that
#   our train loss goes from 1.xxx to 0.xxx 
#   Maybe you can try 1e-1 and see what happens 
#   lr=0.005 is a little bit low
#   
#
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8, weight_decay=1e-2)
# optimizer = optim.Adam(model.parameters(), lr=0.001) 

#-----------------------------------------------------------------------------
# scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=20, max_epochs=200)
# step_size=5 , gamma = 0.1 : every 5 epoch, lr gets smaller than 0.1 times
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  

#-----------------------------------------------------------------------------
# OneCycleLR - COSINE scheduler
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=10, epochs=num_epochs)  


trained_model = train_model(model, criterion, optimizer,
                            dataloaders, device, dataset_sizes, 
                            scheduler, num_epochs)      

trained_model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'input_size': 600,
              'output_size': 416,
              'epochs': num_epochs,
              'batch_size': BATCH_SIZE,
              'trained_model': trained_model,
              'optimizer': optimizer.state_dict(),
              'state_dict': trained_model.state_dict(),
              'class_to_idx': trained_model.class_to_idx
              }
ckpt_path = "ckpt" ; f_name = 'food_class_hospital.pth'
torch.save(checkpoint, os.path.join(ckpt_path, f_name))


                    

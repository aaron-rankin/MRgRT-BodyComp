"""
Main Training script 
"""
import numpy as np
from albumentations.augmentations.geometric.transforms import ElasticTransform
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import optimizer

from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim

from configparser import ConfigParser
from argparse import ArgumentParser

from utils.Dataset import customDataset
from utils.Loops import segmenter
from utils.Writer import customWriter
from utils.Models import Titan_base, Titan_vit
import cv2
import os

import segmentation_models_pytorch as smp




#~ Arg. Parse for config file
parser = ArgumentParser(description='Inference for testing segmentation model')
parser.add_argument('--config', '--c', dest='config',
                    default='config.ini', type=str, help='Path to config.ini file')
args = parser.parse_args()


#~ ====  Read config === 
config  = ConfigParser(allow_no_value= True)

config.read(args.config)

root_dir = config['DIRECTORIES']['InputDirectory']
logdir = config['DIRECTORIES']['OutputDirectory']


#~ PARAMS
batch_size = int(config['TRAINING']['BatchSize'])
num_epochs= int(config['TRAINING']['NumEpochs'])
inputChannels = int(config['TRAINING']['InputChannels'])
outputClasses = int(config['TRAINING']['OutputClasses'])
learning_rate = float(config['TRAINING']['LR'])
device = f"cuda:{int(config['TRAINING']['GPU'])}"
print('Using device: ', device)
torch.cuda.set_device(device)
weight_path = config['TRAINING']['InitWeights']
input_size = int(config['TRAINING']['InputSize'])


def load_weights(model, pt_model):
    ##
    #* Load weights
    
    pt_dict = torch.load(pt_model, map_location=device)
    model_dict = model.state_dict()

    pretrained_dict = {}
    #~ In case loading from different architecture
    for key, val in pt_dict.items():
        if key in model_dict:
            if val.shape == model_dict[key].shape:
                pretrained_dict[key] = val
            else:
                print("Shapes don't match")
        else:
            print("Key not in dict")
            continue
    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # Load new state dict
    model.load_state_dict(model_dict)
    return model

def main():
    #* Define augmentations
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5, limit=20, border_mode=cv2.BORDER_CONSTANT, value=0),
        #A.GridDistortion(p=0.5, num_steps=3, distort_limit=0.3, border_mode=4, interpolation=1),
        A.RandomScale(),
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(
            0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensorV2(transpose_mask=True)
    ])
    #* Normalise to ImageNet mean and std 
    valid_transforms = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
        max_pixel_value=1), 
        ToTensorV2(transpose_mask=True)
    ])

    folds = np.arange(5)
    num_folds = len(folds)

    for i in range(num_folds):


    #*===== INITIALISE MODEL =========
    
        if weight_path != "None":
            #model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=outputClasses)
            #model = Titan_base(num_classes=outputClasses)
            #model = TITAN_vit(inputChannels=inputChannels, outputChannels=outputClasses)
            encoder='tu-mixnet_m' 
            model = Titan_vit(inputChannels=inputChannels, num_classes=outputClasses, encoder=encoder)
            model = load_weights(model, weight_path)
            model.to(device)
            preprocessing_fn = None
        else:
            #encoder = 'tu-efficientnetv2_s' - trains very fast
            #encoder = 'tu-seresnet34'
            encoder='tu-mixnet_m' ## Seems like best option
            #encoder='tu-mixnet_s'
            model = Titan_vit(inputChannels=inputChannels, num_classes=outputClasses, encoder=encoder)#.model
            model.to(device)
            preprocessing_fn = None #smp.encoders.get_preprocessing_fn('resnet18', pretrained='imagenet')
        #*============================

        test_fold = folds[i]
        val_fold = folds[(i + 1) % num_folds]
        train_folds = np.delete(folds, [i, (i + 1) % num_folds])

        #~ init. Datasets
        print('====================='*3)
        print('Input Directory: ', root_dir)
        print(f'Output Directory: {config["DIRECTORIES"]["OutputDirectory"]}')
        print('====================='*3)
        
        print(f'Starting Training | Training Folds: {train_folds} | Val Fold: {val_fold} | Test Fold: {test_fold}')

        train_dataset = customDataset(root_dir, folds = train_folds, transforms=train_transforms,
                                      read_masks=True, normalise=config['TRAINING'].getboolean('Normalise'),
                                      window=config['TRAINING'].getint('Window') if config['TRAINING']['Window'] != '' else None,
                                      level=config['TRAINING'].getint('Level') if config['TRAINING']['Level'] != '' else None,
                                      input_channels=outputClasses,
                                      preprocessing=preprocessing_fn)

        valid_dataset = customDataset(root_dir, folds = [val_fold], transforms=valid_transforms,
                                      read_masks=True, normalise=config['TRAINING'].getboolean('Normalise'),
                                      window=config['TRAINING'].getint('Window') if config['TRAINING']['Window'] != '' else None,
                                      level=config['TRAINING'].getint('Level') if config['TRAINING']['Level'] != '' else None,
                                      input_channels=outputClasses,
                                      preprocessing=preprocessing_fn)
        
        print('TRAINING/VALIDATION', train_dataset.__len__(), valid_dataset.__len__())
        train_loader = DataLoader(train_dataset, batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        writer = customWriter(logdir, batch_size)

        #~ ==== TRAIN =====
        seg = segmenter(model, optimizer, train_loader,
                        valid_loader, writer, num_epochs, num_outputs=outputClasses,
                        device=device, output_path=logdir, test_fold=i)
        seg.forward()


    print('====================='*2)
    print('Training Complete')
if __name__ == '__main__':
    main()

"""
Check model inference.

#! Fold i - refers to the fold that is withheld for testing
#! Run inference on fold i and save predictions
#! Fold i+1 - refers to the fold that is used for validation
"""
from typing import OrderedDict
from utils.Dataset import customDataset
from utils.Loops import segmenter

import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from torchvision.models.segmentation import fcn_resnet101
from utils.Models import Titan_vit

from configparser import ConfigParser
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser(description='Inference for testing segmentation model')
parser.add_argument('--config', '--c', dest='config', 
    default='config.ini', type=str, help='Path to config.ini file')
args = parser.parse_args()

#~ === CONFIG ===
config = ConfigParser()
config.read(args.config)

test_path = config['INFERENCE']['inputPath']
batch_size=1
outputClasses = int(config['TRAINING']['OutputClasses'])
device = f"cuda:{int(config['TRAINING']['GPU'])}"
print('Using device: ', device)
torch.cuda.set_device(device)

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
                continue
        else:

            print(key)
            print("Key not in dict")
            continue
    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # Load new state dict
    model.load_state_dict(model_dict)
    return model

def inference(model, test_loader):
    #~ Inference loop
    id_list = []
    for idx, data in enumerate(tqdm(test_loader)):
        inputs, id_ = data['inputs'].to(
            device, dtype=torch.float32), data['id']
        id_list.append(id_)
        outputs = model(inputs)

        inputs = inputs.cpu().numpy()
        if isinstance(outputs, OrderedDict):
            outputs = outputs['out']
        outputs = outputs.cpu().detach().numpy()

        
        if idx == 0:
            input_slices = inputs
            predictions = outputs
        else:
            input_slices = np.concatenate((input_slices, inputs), axis=0)
            predictions = np.concatenate((predictions, outputs), axis=0)
    return input_slices, predictions, id_list


def main():
    #* Transforms
    test_transforms = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(
        0.229, 0.224, 0.225), max_pixel_value=1), ToTensorV2(transpose_mask=True)])

    folds = np.arange(5)
    for i in folds:
        print('Evuating fold:', i)
        test_dataset = customDataset(
            test_path, [i], test_transforms, read_masks=False,
            normalise=config['TRAINING'].getboolean('Normalise'),
            window=config['TRAINING'].getint('Window') if config['TRAINING']['Window'] != '' else None,
            level=config['TRAINING'].getint('Level') if config['TRAINING']['Level'] != '' else None,
            worldmatch=False)
        
        print('Test Size', test_dataset.__len__())

        test_generator = DataLoader(test_dataset, batch_size)

        weights_path = os.path.join(config['DIRECTORIES']['OutputDirectory'], f'fold_{i}', 'best_model.pt')
        print('Loading weights from:', weights_path)
        #* Load model
        if config['TRAINING']['InitWeights'] != "None":
            model = Titan_vit(num_classes=outputClasses)
            model = load_weights(model, weights_path)
            model.to(device)
        else:
            model=Titan_vit(num_classes=4)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)

        model.eval()

        #~ ==== INFERENCE =====
        slices, preds, ids = inference(model, test_generator)
        np.savez(os.path.join(config['DIRECTORIES']['OutputDirectory'], f'fold_{i}', 'predictions.npz'),
                 slices=slices, masks=preds, id=ids)

if __name__=='__main__':
    main()
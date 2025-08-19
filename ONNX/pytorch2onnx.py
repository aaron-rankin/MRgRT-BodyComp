"""
Script for converting trained models to ONNX.

Applies optimization + quantization?
"""

import os
import numpy as np
import torch

from configparser import ConfigParser
from argparse import ArgumentParser

import onnx_utils as ox

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

device = 'cpu'

#~ Arg. Parse for config file
parser = ArgumentParser(description='Inference for testing segmentation model')
parser.add_argument('--config', '--c', dest='config',
                    default='config.ini', type=str, help='Path to config.ini file')
args = parser.parse_args()

config = ConfigParser()
config.read(args.config)

onnx_config = config['ONNX']
numOutputs = int(config['TRAINING']['OutputClasses'])


def main():
    #~ Load models
    if onnx_config['ModelSource'] == 'torchvision':
        model = ox.get_torchvision_model(onnx_config['Architecture'], int(config['TRAINING']['OutputClasses']))
        model.load_state_dict(torch.load(onnx_config['Path2weights'], map_location=torch.device(device)))

    elif onnx_config['ModelSource'] == 'segmentation_models_pytorch':
        model = ox.get_segmentation_models_model(onnx_config['Architecture'], 
            int(config['TRAINING']['OutputClasses']), int(config['TRAINING']['InputChannels']))
        model.load_state_dict(torch.load(onnx_config['Path2weights'], map_location=torch.device(device)))
    
    elif onnx_config['ModelSource'] == 'custom':
        model = ox.get_custom_model(onnx_config['Architecture'], num_classes=numOutputs)
        if model is None:
            raise NotImplementedError('Architecture not implemented yet.')
        model.load_state_dict(torch.load(onnx_config['Path2weights'], map_location=torch.device(device)))
    
    else:
        raise ValueError("Choose architecture from [torchvision/segmentation_models_pytorch/custom]")
    
    #* Template input tensor
    dummy_input = torch.randn(1, int(config['TRAINING']['InputChannels']), 
        int(config['TRAINING']['InputSize']), int(config['TRAINING']['InputSize']), requires_grad=True, device=device)
    print(dummy_input.shape)
    #* Export with dynamic axes (batch, height, width)
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'width', 3: 'height'},
        'output': {0: 'batch_size', 2: 'width', 3: 'height'}
        }

    #* Inference with original model, to compare performance
    model.eval()
    output_filepath = os.path.join(config['DIRECTORIES']['OutputDirectory'], onnx_config['OutputFilename'])
    torch.onnx.export(model, dummy_input, output_filepath, 
            input_names=['input'], output_names=['output'], export_params=True, 
            opset_version=13, dynamic_axes=dynamic_axes, do_constant_folding=True, 
            verbose=False)

    #* Quantise...
    if onnx_config.getboolean('Quantise'):
        print('Attempting to quantise ONNX model...')
        quant_name = onnx_config['OutputFilename'].split('.')[0] + '.quant.onnx'
        onnx_modelpath = os.path.join(config['DIRECTORIES']['OutputDirectory'], quant_name)
        quantize_dynamic(output_filepath, onnx_modelpath, weight_type=QuantType.QUInt8)

        #* Check model is valid
        onnx_model = onnx.load(onnx_modelpath)
        onnx.checker.check_model(onnx_model)
        print('Model succesfully checked.')

        #* Check model predictions
        ox.check_predictions(onnx_modelpath, model, dummy_input)
    else:
        #* Check model is valid
        onnx_model = onnx.load(output_filepath)
        onnx.checker.check_model(onnx_model)
        print('Model succesfully checked.')

        #* Check model predictions
        ox.check_predictions(onnx_model, model, dummy_input)


if __name__=='__main__':
    main()
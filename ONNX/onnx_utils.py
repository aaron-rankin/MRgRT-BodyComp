"""
Useful functions for onnx conversion
"""
from typing import OrderedDict
import torchvision.models.segmentation as tvms
import segmentation_models_pytorch as smpt
import torch
import onnxruntime
import numpy as np
import sys

sys.path.append("/home/donal/segmentation_workflow/")
from utils.Models import Titan_base, Titan_vit


#~ ======== Get models
def get_torchvision_model(name, classes):
    """
    Figure out which case we have and load the model & weights
    """

    return tvms.__dict__[name](num_classes=classes)

def get_segmentation_models_model(name, classes, in_channels):
    print(smpt.__dict__)
    return smpt.__dict__[name](classes=classes, in_channels=in_channels, encoder_weights=None)

def get_custom_model(name, **kwargs):
    print('Collecting ', name)
    model_bank = {'Titan_base': Titan_base(**kwargs),
                  'Titan_vit': Titan_vit(**kwargs)}
    return model_bank[name] if name in model_bank else None

def check_predictions(onnx_modelpath, pytorch_model, dummy_input):
    #~ Check prediction 
    torch_out = pytorch_model(dummy_input)

    ort_session = onnxruntime.InferenceSession(onnx_modelpath)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    if isinstance(ort_outs, OrderedDict):
        torch_pred = np.round(to_numpy(sigmoid(torch_out['out'])))
        onnx_pred = np.round(to_numpy(sigmoid(torch.tensor(ort_outs[0]))))
        np.testing.assert_allclose(torch_pred, onnx_pred, rtol=1e-03, atol=1e-05)
    else:
        torch_pred = to_numpy(torch.argmax(torch_out, dim=0))
        onnx_pred = np.argmax(ort_outs[0], axis=0)
        np.testing.assert_allclose(torch_pred, onnx_pred, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")